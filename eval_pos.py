#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt


UPOS_SET = {
    "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM",
    "PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"
}

# XPOS helpers for English (Penn Treebank style)
XPOS_ADJ = {"JJ","JJR","JJS"}
XPOS_NOUN = {"NN","NNS"}
XPOS_PROPN = {"NNP","NNPS"}

def detect_format(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    if {"xpos_tags"} & cols or {"upos_tags"} & cols:
        return "sentence"
    if {"sentence_id","token_id","xpos"} <= cols or {"sentence_id","token_id","upos"} <= cols:
        return "token"
    if "text" in cols:
        return "sentence"
    return "token"

def detect_tagset(tags: List[str]) -> str:
    sample = [t for t in tags if t and t != "_" ]
    if not sample:
        return "xpos"
    upos_like = sum(1 for t in sample if t in UPOS_SET)
    return "upos" if upos_like >= len(sample) * 0.6 else "xpos"

def choose_tag_column(cols: List[str], priority: str) -> Optional[str]:
    cset = set(c.lower() for c in cols)
    candidates = []
    if "xpos_tags" in cset: candidates.append("xpos_tags")
    if "upos_tags" in cset: candidates.append("upos_tags")
    if not candidates:
        return None
    if priority.lower() == "xpos" and "xpos_tags" in candidates:
        return "xpos_tags"
    if priority.lower() == "upos" and "upos_tags" in candidates:
        return "upos_tags"
    return candidates[0]

def normalize_series_to_str_list(seq: List[Optional[str]]) -> List[str]:
    return [("_" if s is None else str(s)) for s in seq]

def split_tag_seq(tag_str: str) -> List[str]:
    tag_str = "" if tag_str is None else str(tag_str)
    tag_str = tag_str.strip()
    return tag_str.split() if tag_str else []

def load_sentence_format(df_gold: pd.DataFrame, df_pred: pd.DataFrame, tag_priority: str):
    join_key = None
    for candidate in ["row_id","sentence_id","id"]:
        if candidate in df_gold.columns and candidate in df_pred.columns:
            join_key = candidate
            break
    if join_key:
        merged = pd.merge(
            df_gold, df_pred,
            on=join_key, how="inner", suffixes=("_gold", "_pred"),
            validate="one_to_one"
        )
    else:
        df_gold = df_gold.reset_index(drop=True).copy()
        df_pred = df_pred.reset_index(drop=True).copy()
        merged = pd.concat([df_gold.add_suffix("_gold"), df_pred.add_suffix("_pred")], axis=1)

    tag_col_gold = choose_tag_column(list(merged.columns), tag_priority + "_gold")
    tag_col_pred = choose_tag_column(list(merged.columns), tag_priority + "_pred")
    if tag_col_gold is None or tag_col_pred is None:
        tag_col_gold = "xpos_tags_gold" if "xpos_tags_gold" in merged.columns else "upos_tags_gold"
        tag_col_pred = "xpos_tags_pred" if "xpos_tags_pred" in merged.columns else "upos_tags_pred"

    if tag_col_gold.endswith("_gold"): tags_gold_col = tag_col_gold
    else: tags_gold_col = tag_col_gold + "_gold"
    if tag_col_pred.endswith("_pred"): tags_pred_col = tag_col_pred
    else: tags_pred_col = tag_col_pred + "_pred"

    text_gold_col = "text_gold" if "text_gold" in merged.columns else None
    text_pred_col = "text_pred" if "text_pred" in merged.columns else None

    gold_seqs, pred_seqs, texts, word_lists = [], [], [], []
    for _, row in merged.iterrows():
        gtags = split_tag_seq(row.get(tags_gold_col, ""))
        ptags = split_tag_seq(row.get(tags_pred_col, ""))
        gold_seqs.append(gtags)
        pred_seqs.append(ptags)
        t_gold = row.get(text_gold_col, None)
        t_pred = row.get(text_pred_col, None)
        text = str(t_gold) if pd.notna(t_gold) else (str(t_pred) if pd.notna(t_pred) else "")
        texts.append(text)
        word_lists.append(text.split() if text else [])
    return gold_seqs, pred_seqs, texts, word_lists, merged

def load_token_format(df_gold: pd.DataFrame, df_pred: pd.DataFrame, tag_priority: str):
    required = ["sentence_id","token_id"]
    for col in required:
        if col not in df_gold.columns or col not in df_pred.columns:
            raise ValueError(f"Token format requires '{col}' in both CSVs.")

    tag_col = "xpos" if (tag_priority.lower()=="xpos" and "xpos" in df_gold.columns and "xpos" in df_pred.columns) \
                    else ("upos" if "upos" in df_gold.columns and "upos" in df_pred.columns else None)
    if tag_col is None:
        if "xpos" in df_gold.columns and "xpos" in df_pred.columns:
            tag_col = "xpos"
        elif "upos" in df_gold.columns and "upos" in df_pred.columns:
            tag_col = "upos"
        else:
            raise ValueError("Could not find common tag column (xpos or upos) in token-level CSVs.")

    merged = pd.merge(
        df_gold[["sentence_id","token_id",tag_col,"word"]].rename(columns={tag_col:"tag_gold","word":"word_gold"}),
        df_pred[["sentence_id","token_id",tag_col,"word"]].rename(columns={tag_col:"tag_pred","word":"word_pred"}),
        on=["sentence_id","token_id"], how="inner", validate="one_to_one"
    ).sort_values(["sentence_id","token_id"])

    gold_seqs, pred_seqs, texts, word_lists = [], [], [], []
    for sid, g in merged.groupby("sentence_id"):
        g = g.sort_values("token_id")
        gold_seqs.append(normalize_series_to_str_list(g["tag_gold"].tolist()))
        pred_seqs.append(normalize_series_to_str_list(g["tag_pred"].tolist()))
        words = g["word_gold"].fillna(g["word_pred"]).astype(str).tolist()
        word_lists.append(words)
        texts.append(" ".join(words))
    return gold_seqs, pred_seqs, texts, word_lists, merged

def flatten(lists: List[List[str]]) -> List[str]:
    return [x for sub in lists for x in sub]

def safe_align_with_words(gold_seqs, pred_seqs, word_lists):
    g2, p2, w2 = [], [], []
    mismatched = 0
    for g, p, w in zip(gold_seqs, pred_seqs, word_lists):
        m = min(len(g), len(p), len(w))
        if len(g) != len(p) or len(g) != len(w) or len(p) != len(w):
            mismatched += 1
        g2.append(g[:m]); p2.append(p[:m]); w2.append(w[:m])
    return g2, p2, w2, mismatched

def tagset_from_data(gold_flat: List[str], priority: str) -> str:
    if priority.lower() in {"upos","xpos"}:
        return priority.lower()
    return detect_tagset(gold_flat)

def coarse_pos(tag: str, tagset: str) -> str:
    if not tag or tag == "_":
        return "_"
    if tagset == "upos":
        return tag if tag in UPOS_SET else "X"
    # xpos (PTB-like for English)
    if tag in XPOS_ADJ: return "ADJ"
    if tag in XPOS_NOUN: return "NOUN"
    if tag in XPOS_PROPN: return "PROPN"
    if tag.startswith("VB"): return "VERB"
    if tag.startswith("RB"): return "ADV"
    if tag in {"IN"}: return "ADP"
    if tag in {"CC"}: return "CCONJ"
    if tag in {"DT","PDT","WDT"}: return "DET"
    if tag in {"PRP","PRP$","WP","WP$","EX"}: return "PRON"
    if tag in {"CD"}: return "NUM"
    if tag in {"TO","RP","POS"}: return "PART"
    if tag in {"UH"}: return "INTJ"
    if tag in {"SYM"}: return "SYM"
    if tag in {"." , ",", ":", "``", "''", "-LRB-", "-RRB-"}: return "PUNCT"
    return "X"

def make_pairs(tags: List[str], tagset: str) -> List[str]:
    coarse = [coarse_pos(t, tagset) for t in tags]
    return [f"{coarse[i]}+{coarse[i+1]}" for i in range(len(coarse)-1)]

def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def plot_confusion(cm: np.ndarray, labels: List[str], out_png: str, title="Confusion Matrix"):
    fig = plt.figure(figsize=(max(6, min(16, len(labels)*0.5)), max(6, min(16, len(labels)*0.5))))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def collect_pair_examples(pair_cat: str,
                          gold_seqs: List[List[str]],
                          pred_seqs: List[List[str]],
                          words: List[List[str]],
                          tagset_family: str,
                          max_examples: int = 5):
    fp, fn = [], []
    for sid, (gtags, ptags, w) in enumerate(zip(gold_seqs, pred_seqs, words)):
        g_pairs = make_pairs(gtags, tagset_family)
        p_pairs = make_pairs(ptags, tagset_family)
        m = min(len(g_pairs), len(p_pairs), max(0, len(w)-1))
        for i in range(m):
            # token-level context
            pair_words = " ".join(w[i:i+2]) if i+1 < len(w) else ""
            gold_pair = g_pairs[i]
            pred_pair = p_pairs[i]
            gold_pair_tags = gtags[i:i+2]
            pred_pair_tags = ptags[i:i+2]
            if pred_pair == pair_cat and gold_pair != pair_cat and len(fp) < max_examples:
                fp.append({
                    "sentence_index": sid,
                    "position": i,
                    "pair_words": pair_words,
                    "gold_pair": gold_pair,
                    "pred_pair": pred_pair,
                    "gold_tags_pair": gold_pair_tags,
                    "pred_tags_pair": pred_pair_tags,
                    "sentence_text": " ".join(w)
                })
            if gold_pair == pair_cat and pred_pair != pair_cat and len(fn) < max_examples:
                fn.append({
                    "sentence_index": sid,
                    "position": i,
                    "pair_words": pair_words,
                    "gold_pair": gold_pair,
                    "pred_pair": pred_pair,
                    "gold_tags_pair": gold_pair_tags,
                    "pred_tags_pair": pred_pair_tags,
                    "sentence_text": " ".join(w)
                })
            if len(fp) >= max_examples and len(fn) >= max_examples:
                return fp, fn
    return fp, fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to gold CSV")
    ap.add_argument("--pred", required=True, help="Path to predicted CSV (same format)")
    ap.add_argument("--format", choices=["auto","sentence","token"], default="auto")
    ap.add_argument("--tag-priority", choices=["auto","xpos","upos"], default="auto",
                    help="Which tag family to prefer when both are present.")
    ap.add_argument("--pair-cats", default="ADJ+NOUN,NOUN+NOUN,PROPN+PROPN,ADJ+PROPN",
                    help="Comma-separated list of pair categories to score (after coarse mapping).")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_gold = pd.read_csv(args.gold)
    df_pred = pd.read_csv(args.pred)

    data_format = args.format if args.format != "auto" else detect_format(df_gold)
    if data_format == "sentence":
        gold_seqs, pred_seqs, texts, word_lists, merged = load_sentence_format(df_gold, df_pred, args.tag_priority)
    else:
        gold_seqs, pred_seqs, texts, word_lists, merged = load_token_format(df_gold, df_pred, args.tag_priority)

    # Align (truncate) tags & words per sentence
    gold_seqs, pred_seqs, word_lists, mismatched = safe_align_with_words(gold_seqs, pred_seqs, word_lists)

    gold_flat = normalize_series_to_str_list(flatten(gold_seqs))
    pred_flat = normalize_series_to_str_list(flatten(pred_seqs))

    tagset_family = tagset_from_data(gold_flat, args.tag_priority)

    # === Global metrics ===
    overall_acc = accuracy_score(gold_flat, pred_flat)
    pr_micro = precision_recall_fscore_support(gold_flat, pred_flat, average="micro", zero_division=0)
    pr_macro = precision_recall_fscore_support(gold_flat, pred_flat, average="macro", zero_division=0)
    pr_weighted = precision_recall_fscore_support(gold_flat, pred_flat, average="weighted", zero_division=0)

    # === Per-class metrics ===
    labels = sorted(list(set(gold_flat) | set(pred_flat)))
    cls_report = classification_report(gold_flat, pred_flat, labels=labels, output_dict=True, zero_division=0)
    per_class = {lbl: cls_report.get(lbl, {}) for lbl in labels}

    # === Confusion matrix ===
    cm = confusion_matrix(gold_flat, pred_flat, labels=labels)
    cm_csv = os.path.join(args.outdir, "confusion_matrix.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_csv)
    cm_png = os.path.join(args.outdir, "confusion_matrix.png")
    plot_confusion(cm, labels, cm_png)

    # === Per-sentence exact match ===
    total_sents = len(gold_seqs)
    exact = 0
    wrong_rows = []
    for i, (g, p, wtxt) in enumerate(zip(gold_seqs, pred_seqs, texts)):
        if g == p:
            exact += 1
        else:
            wrong_rows.append({
                "row_index": i,
                "text": wtxt,
                "gold_tags": " ".join(g),
                "pred_tags": " ".join(p),
                "tokens": len(g),
                "num_wrong": sum(1 for a,b in zip(g,p) if a!=b)
            })
    sentence_exact_acc = exact / total_sents if total_sents>0 else 0.0
    wrong_csv = os.path.join(args.outdir, "wrong_sentences.csv")
    pd.DataFrame(wrong_rows).to_csv(wrong_csv, index=False)

    # === Pairwise (keyphrase-like) metrics ===
    pair_cats = [x.strip() for x in args.pair_cats.split(",") if x.strip()]
    counts = {c: {"tp":0,"fp":0,"fn":0} for c in pair_cats}
    for gtags, ptags in zip(gold_seqs, pred_seqs):
        g_pairs = make_pairs(gtags, tagset_family)
        p_pairs = make_pairs(ptags, tagset_family)
        m = min(len(g_pairs), len(p_pairs))
        for i in range(m):
            gp = g_pairs[i]; pp = p_pairs[i]
            for cat in pair_cats:
                if gp == cat and pp == cat:
                    counts[cat]["tp"] += 1
                elif gp != cat and pp == cat:
                    counts[cat]["fp"] += 1
                elif gp == cat and pp != cat:
                    counts[cat]["fn"] += 1

    pair_metrics = {}
    for cat, d in counts.items():
        pair_metrics[cat] = {**d, **prf_from_counts(d["tp"], d["fp"], d["fn"])}

    # === Collect FP/FN examples for ADJ+NOUN (and any requested pairs) ===
    pair_examples = {}
    for cat in pair_cats:
        fp, fn = collect_pair_examples(cat, gold_seqs, pred_seqs, word_lists, tagset_family, max_examples=5)
        pair_examples[cat] = {"false_positives": fp, "false_negatives": fn}

    # === Per-class report CSV ===
    per_class_rows = []
    for lbl in labels:
        r = per_class.get(lbl, {})
        per_class_rows.append({
            "label": lbl,
            "precision": r.get("precision", 0.0),
            "recall": r.get("recall", 0.0),
            "f1": r.get("f1-score", 0.0),
            "support": r.get("support", 0)
        })
    per_class_csv = os.path.join(args.outdir, "per_class_metrics.csv")
    pd.DataFrame(per_class_rows).to_csv(per_class_csv, index=False)

    # === Notes to clarify global metrics (micro vs. accuracy, etc.) ===
    metric_notes = {
        "accuracy": "Token-level accuracy: percent of tokens whose tag matches gold.",
        "micro": "Micro-averaged over tokens. For single-label token classification, micro precision/recall/F1 ≈ accuracy.",
        "macro": "Macro-averaged across labels (each label equal weight). Sensitive to rare labels.",
        "weighted": "Weighted average across labels (weights = support)."
    }

    # === Summary JSON for report ===
    summary = {
        "format": data_format,
        "tagset_family": tagset_family,
        "num_sentences": total_sents,
        "num_tokens": len(gold_flat),
        "mismatched_sentence_lengths": mismatched,
        "global": {
            "accuracy": overall_acc,
            "precision_micro": pr_micro[0],
            "recall_micro": pr_micro[1],
            "f1_micro": pr_micro[2],
            "precision_macro": pr_macro[0],
            "recall_macro": pr_macro[1],
            "f1_macro": pr_macro[2],
            "precision_weighted": pr_weighted[0],
            "recall_weighted": pr_weighted[1],
            "f1_weighted": pr_weighted[2],
        },
        "metric_notes": metric_notes,
        "per_sentence": {
            "exact_match_accuracy": sentence_exact_acc,
            "num_exact": exact,
            "num_sentences": total_sents
        },
        "labels": labels,
        "confusion_matrix_csv": cm_csv,
        "confusion_matrix_png": cm_png,
        "per_class_csv": per_class_csv,
        "wrong_sentences_csv": wrong_csv,
        "pairs": pair_metrics,
        "pair_examples": pair_examples
    }
    metrics_json = os.path.join(args.outdir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Tokens: {len(gold_flat)} | Sentences: {total_sents}")
    print(f"[OK] Accuracy: {overall_acc:.4f} | Sentence exact-match: {sentence_exact_acc:.4f}")
    print(f"[OK] Confusion matrix -> {cm_csv} / {cm_png}")
    print(f"[OK] Per-class -> {per_class_csv}")
    print(f"[OK] Wrong sentences -> {wrong_csv}")
    print(f"[OK] Summary JSON -> {metrics_json}")

if __name__ == "__main__":
    main()
