#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Flair POS tagging over a CSV and save predictions in the same data format.

Supported input formats (auto-detected):

A) Sentence-level (one row per sentence)
    columns: [row_id?], text, [xpos_tags?], [upos_tags?]
    -> produces: xpos_tags (pred) and upos_tags (pred)

B) Token-level (one row per token)
    columns: sentence_id, token_id, word, [xpos?], [upos?]
    -> produces: sentence_id, token_id, word, xpos (pred) and upos (pred)

Notes:
- By default we PRESERVE your whitespace tokenization to keep alignment with gold.
  (We build Flair Sentence objects from your tokens with identical boundaries.)
- Flair models:
      'pos' / 'pos-fast'     -> XPOS (Penn Treebank-style, e.g., JJ, NN, VBZ)
      'upos' / 'upos-fast'   -> UPOS (Universal POS, e.g., ADJ, NOUN, VERB)
- If you prefer Flair's own tokenizer (may break alignment), pass --use-flair-tokenizer.

Usage:
    python predict_flair_pos.py --in gold.csv --out pred.csv
    # token-level input:
    python predict_flair_pos.py --in gold_tok.csv --out pred_tok.csv --format token
    # specify models:
    python predict_flair_pos.py --in gold.csv --out pred.csv --xpos-model pos-fast --upos-model upos-fast
"""

import argparse
import sys
from typing import List, Tuple, Optional
import pandas as pd

# ----------------------------
# Utilities
# ----------------------------

def detect_format(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if {"sentence_id", "token_id", "word"} <= cols:
        return "token"
    if "text" in cols:
        return "sentence"
    return "sentence"

def _ensure_flair_tagger(model_name: str):
    try:
        from flair.models import SequenceTagger
    except Exception as e:
        print("[ERROR] Flair is not installed. Install with: pip install flair", file=sys.stderr)
        sys.exit(2)
    try:
        return SequenceTagger.load(model_name)
    except Exception as e:
        print(f"[ERROR] Could not load Flair model '{model_name}'. "
              f"Make sure the name is correct (e.g., 'upos-fast', 'upos', 'pos-fast', 'pos').\nReason: {e}",
              file=sys.stderr)
        sys.exit(2)

def _make_sentence_from_tokens(tokens: List[str]):
    from flair.data import Sentence
    text = " ".join(tokens)
    return Sentence(text, use_tokenizer=False)  # preserves your whitespace tokens


def _make_sentence_from_text(text: str):
    from flair.data import Sentence
    return Sentence(text or "")  # default use_tokenizer=True

def _predict_flair_tags(
    sentences,
    xpos_tagger=None,
    upos_tagger=None,
    mini_batch_size: int = 32
):
    """
    Populate sentences with predicted tags from the provided taggers.
    Returns (xpos_tags_per_sent, upos_tags_per_sent).
    Compatible with Flair >= 0.11 (no get_tag()).
    """
    if xpos_tagger is not None:
        xpos_tagger.predict(sentences, mini_batch_size=mini_batch_size, verbose=False)
    if upos_tagger is not None:
        upos_tagger.predict(sentences, mini_batch_size=mini_batch_size, verbose=False)

    def labels_for(sent, tagger):
        if tagger is None:
            return ["_"] * len(sent)
        tag_type = tagger.tag_type  # e.g., "pos" or "upos"
        seq = []
        for tok in sent:
            # get_labels returns a list[Label]; pick first if present
            labs = tok.get_labels(tag_type)
            seq.append(labs[0].value if labs else "_")
        return seq

    xpos_all = [labels_for(s, xpos_tagger) for s in sentences]
    upos_all = [labels_for(s, upos_tagger) for s in sentences]
    return xpos_all, upos_all


# ----------------------------
# Sentence-level
# ----------------------------

def run_sentence_mode(
    df: pd.DataFrame,
    xpos_tagger,
    upos_tagger,
    use_flair_tok: bool,
    batch_size: int
) -> pd.DataFrame:
    out = df.copy()
    texts = out["text"].fillna("").astype(str).tolist()

    # Prepare Sentence objects, preserving tokenization unless told otherwise
    if use_flair_tok:
        sentences = [_make_sentence_from_text(t) for t in texts]
    else:
        token_seqs = [t.split() if t else [] for t in texts]
        sentences = [_make_sentence_from_tokens(toks) for toks in token_seqs]

    xpos_pred_seqs, upos_pred_seqs = _predict_flair_tags(
        sentences,
        xpos_tagger=xpos_tagger,
        upos_tagger=upos_tagger,
        mini_batch_size=batch_size
    )

    out["xpos_tags"] = [" ".join(seq) for seq in xpos_pred_seqs]
    out["upos_tags"] = [" ".join(seq) for seq in upos_pred_seqs]
    return out

# ----------------------------
# Token-level
# ----------------------------

def run_token_mode(
    df: pd.DataFrame,
    xpos_tagger,
    upos_tagger,
    use_flair_tok: bool,
    batch_size: int
) -> pd.DataFrame:
    required = ["sentence_id", "token_id", "word"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Token format requires column '{col}'.")

    # Group rows by sentence, preserve token order
    grouped = []
    sentence_orders = []  # keep (sid, indices) to reconstruct rows
    for sid, g in df.groupby("sentence_id", sort=True):
        g_sorted = g.sort_values("token_id")
        words = g_sorted["word"].astype(str).tolist()
        sentence_orders.append((sid, g_sorted.index.tolist()))
        if use_flair_tok:
            # Let Flair re-tokenize (alignment may change!)
            text = " ".join(words)
            grouped.append(_make_sentence_from_text(text))
        else:
            grouped.append(_make_sentence_from_tokens(words))

    # Predict
    xpos_pred_seqs, upos_pred_seqs = _predict_flair_tags(
        grouped,
        xpos_tagger=xpos_tagger,
        upos_tagger=upos_tagger,
        mini_batch_size=batch_size
    )

    # Reconstruct rows
    records = []
    for (sid, idx_list), xpos_seq, upos_seq in zip(sentence_orders, xpos_pred_seqs, upos_pred_seqs):
        # If tokenized by Flair, lengths may differ; be safe.
        m = min(len(idx_list), len(xpos_seq), len(upos_seq))
        for i in range(m):
            row = df.loc[idx_list[i]]
            records.append({
                "sentence_id": row["sentence_id"],
                "token_id": row["token_id"],
                "word": row["word"],
                "xpos": xpos_seq[i],
                "upos": upos_seq[i]
            })

    cols = ["sentence_id", "token_id", "word", "xpos", "upos"]
    return pd.DataFrame.from_records(records, columns=cols)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (predictions)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto",
                    help="Force input format (default: auto)")
    ap.add_argument("--xpos-model", default="pos-fast",
                    help="Flair model for XPOS (e.g., pos, pos-fast)")
    ap.add_argument("--upos-model", default="upos-fast",
                    help="Flair model for UPOS (e.g., upos, upos-fast)")
    ap.add_argument("--batch-size", type=int, default=32, help="Flair mini-batch size")
    ap.add_argument("--use-flair-tokenizer", action="store_true",
                    help="Use Flair's tokenizer instead of preserving your whitespace tokens (may break alignment).")
    args = ap.parse_args()

    try:
        # Load CSV
        df = pd.read_csv(args.inp)

        # Load taggers
        xpos_tagger = _ensure_flair_tagger(args.xpos_model) if args.xpos_model else None
        upos_tagger = _ensure_flair_tagger(args.upos_model) if args.upos_model else None
        if xpos_tagger is None and upos_tagger is None:
            raise ValueError("At least one of --xpos-model or --upos-model must be provided.")

        # Determine format
        fmt = args.format if args.format != "auto" else detect_format(df)

        # Run
        if fmt == "sentence":
            pred = run_sentence_mode(
                df,
                xpos_tagger=xpos_tagger,
                upos_tagger=upos_tagger,
                use_flair_tok=args.use_flair_tokenizer,
                batch_size=args.batch_size
            )
        else:
            pred = run_token_mode(
                df,
                xpos_tagger=xpos_tagger,
                upos_tagger=upos_tagger,
                use_flair_tok=args.use_flair_tokenizer,
                batch_size=args.batch_size
            )

        # Save
        pred.to_csv(args.out, index=False)
        used_models = []
        if xpos_tagger: used_models.append(f"xpos={args.xpos_model}")
        if upos_tagger: used_models.append(f"upos={args.upos_model}")
        print(f"[OK] Wrote predictions to {args.out} (format={fmt}, models: {', '.join(used_models)})")

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()