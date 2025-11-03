#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust single-process TreeTagger predictions → CSV with the same schema as your gold.

- Auto-detects sentence vs token CSV.
- Preserves whitespace tokenization.
- Streams ALL tokens through ONE TreeTagger process.
- Parses flat output (no reliance on blank lines); splits back to sentences by known lengths.
- Auto-retries with safer args if the first run yields too few lines.
- Optional debug: save raw TT output head to a file.

Usage (examples):
  python predict_treetagger_pos.py \
    --in data.csv \
    --out tree_tagger/preds.csv \
    --tt-cmd /opt/treetagger/bin/tree-tagger \
    --tt-params /opt/treetagger/lib/english.par \
    --tt-args "-quiet"

  # Wrapper (params implicit):
  python predict_treetagger_pos.py \
    --in data.csv \
    --out tree_tagger/preds.csv \
    --tt-cmd /opt/treetagger/bin/tree-tagger-english
"""

import argparse
import sys
import shlex
import subprocess
from typing import List, Tuple, Dict
import pandas as pd

# ---- PTB/XPOS -> UPOS coarse mapping (English) ----
PTB2UPOS: Dict[str, str] = {
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "IN": "ADP",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "WRB": "ADV",
    "CC": "CCONJ",
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    "NN": "NOUN", "NNS": "NOUN",
    "NP": "PROPN", "NPS": "PROPN", "NNP": "PROPN", "NNPS": "PROPN",
    "CD": "NUM",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON", "EX": "PRON",
    "PP": "PRON", "PP$": "PRON",
    "TO": "PART", "RP": "PART", "POS": "PART",
    "MD": "VERB", "VB": "VERB", "VBD": "VERB", "VBG": "VERB",
    "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
    "UH": "INTJ",
    "SYM": "SYM",
    ".": "PUNCT", ",": "PUNCT", ":": "PUNCT", "``": "PUNCT", "''": "PUNCT",
    "-LRB-": "PUNCT", "-RRB-": "PUNCT", "SENT": "PUNCT",
}
TT2PTB = {
    "PP": "PRP",
    "PP$": "PRP$",
    "NP": "NNP",
    "NPS": "NNPS",
    "SENT": ".",   # sentence-final punctuation
}

def normalize_tt_xpos(tag: str) -> str:
    return TT2PTB.get(tag, tag)

def tag_to_upos(tag: str) -> str:
    if not tag or tag == "_":
        return "X"
    return PTB2UPOS.get(tag, "X")

# ---- CSV handling ----
def detect_format(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if {"sentence_id", "token_id", "word"} <= cols:
        return "token"
    if "text" in cols:
        return "sentence"
    return "sentence"

def build_sentences_from_sentence_csv(df: pd.DataFrame) -> List[List[str]]:
    texts = df["text"].fillna("").astype(str).tolist()
    return [t.split() if t else [] for t in texts]

def build_sentences_from_token_csv(df: pd.DataFrame) -> Tuple[List[List[str]], pd.DataFrame]:
    sents: List[List[str]] = []
    ordered = []
    for sid, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        sents.append(g["word"].astype(str).tolist())
        ordered.append(g)
    merged = pd.concat(ordered, axis=0) if ordered else df.head(0)
    return sents, merged

def sentences_to_treetagger_input(sents: List[List[str]]) -> str:
    # One token per line; a blank line after each sentence is harmless.
    lines = []
    for toks in sents:
        lines.extend(toks)
        lines.append("")  # boundary
    return "\n".join(lines).rstrip() + ("\n" if lines else "")

# ---- TreeTagger I/O ----
def run_treetagger(tt_cmd: str, tt_params: str, tt_args_str: str, inp: str, encoding="utf-8") -> Tuple[str, str, int]:
    cmd: List[str] = shlex.split(tt_cmd)
    if tt_args_str:
        cmd += shlex.split(tt_args_str)
    if tt_params:
        cmd.append(tt_params)
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding=encoding,
        )
    except FileNotFoundError:
        raise RuntimeError(f"TreeTagger command not found: {tt_cmd}")
    except Exception as e:
        raise RuntimeError(f"Failed to start TreeTagger: {e}")
    out, err = proc.communicate(inp)
    return out, err, proc.returncode

def parse_treetagger_output_flat(output: str):
    """
    Return a flat list of (token, POS, lemma) triples.
    Flexible: tabs or spaces; 2 or 3 fields.
    """
    triples = []
    for line in output.splitlines():
        line = line.strip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
            if len(parts) < 2:
                continue
        tok = parts[0]
        pos = parts[1] if len(parts) > 1 else "_"
        lem = parts[2] if len(parts) > 2 else "_"
        triples.append((tok, pos, lem))
    return triples

def split_flat_by_lengths(triples, sent_lens):
    sents = []
    i = 0
    n = len(triples)
    for L in sent_lens:
        j = min(i + L, n)
        sents.append(triples[i:j])
        i = j
    return sents

def align_tags_to_input(sents_tokens: List[List[str]],
                        sents_tagged: List[List[Tuple[str, str, str]]]) -> Tuple[List[List[str]], List[List[str]]]:
    xpos_out: List[List[str]] = []
    upos_out: List[List[str]] = []
    for toks, tagged in zip(sents_tokens, sents_tagged):
        m = min(len(toks), len(tagged))
        xseq, useq = [], []
        for i in range(m):
            _, pos, _ = tagged[i]
            norm = normalize_tt_xpos(pos) if pos else "_"
            xseq.append(norm)
            useq.append(tag_to_upos(pos))
        xpos_out.append(xseq)
        upos_out.append(useq)
    return xpos_out, upos_out

# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (predictions)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto")
    ap.add_argument("--tt-cmd", required=True, help="TreeTagger command/path (e.g., '/opt/treetagger/bin/tree-tagger' or 'tree-tagger-english')")
    ap.add_argument("--tt-params", default=None, help="Parameter file path (e.g., '/opt/treetagger/lib/english.par'). Omit for language wrappers.")
    ap.add_argument("--tt-args", default="", help="Extra TreeTagger args (e.g., '-quiet'). Use single-dash flags.")
    ap.add_argument("--no-upos", action="store_true", help="Do not output UPOS (only XPOS).")
    ap.add_argument("--encoding", default="utf-8", help="File/stdio encoding")
    ap.add_argument("--debug-raw-out", default=None, help="Write first ~200 lines of raw TreeTagger output to this file.")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)

    # Build input token sequences
    if fmt == "sentence":
        sents_tokens = build_sentences_from_sentence_csv(df)
    else:
        sents_tokens, _ = build_sentences_from_token_csv(df)

    tt_inp = sentences_to_treetagger_input(sents_tokens)
    total_tokens = sum(len(x) for x in sents_tokens)

    # ---- First attempt: as requested ----
    out1, err1, rc1 = run_treetagger(args.tt_cmd, args.tt_params, args.tt_args, tt_inp, encoding=args.encoding)
    triples1 = parse_treetagger_output_flat(out1)

    # Decide if output looks sane; otherwise retry with safer defaults
    need_retry = (rc1 != 0) or (len(triples1) < max(1, int(0.8 * total_tokens)))
    used_args = args.tt_args

    out_raw = out1
    err_raw = err1
    triples = triples1

    # ---- Retry #1: append -token if not present ----
    if need_retry:
        if "-token" not in (args.tt_args or ""):
            retry_args = (args.tt_args + " -token").strip()
        else:
            retry_args = args.tt_args
        out2, err2, rc2 = run_treetagger(args.tt_cmd, args.tt_params, retry_args, tt_inp, encoding=args.encoding)
        triples2 = parse_treetagger_output_flat(out2)

        if (rc2 == 0) and (len(triples2) >= len(triples)):
            used_args = retry_args
            out_raw, err_raw, triples = out2, err2, triples2
            need_retry = len(triples) < max(1, int(0.8 * total_tokens))
        else:
            # keep earlier best
            pass

    # ---- Retry #2: no extra args at all ----
    if need_retry and args.tt_args:
        out3, err3, rc3 = run_treetagger(args.tt_cmd, args.tt_params, "", tt_inp, encoding=args.encoding)
        triples3 = parse_treetagger_output_flat(out3)
        if (rc3 == 0) and (len(triples3) >= len(triples)):
            used_args = ""
            out_raw, err_raw, triples = out3, err3, triples3

    # Optional: dump raw output (head) for debugging
    if args.debug_raw_out:
        try:
            with open(args.debug_raw_out, "w", encoding=args.encoding) as f:
                lines = out_raw.splitlines()
                head = "\n".join(lines[:200])
                f.write(head + ("\n... (truncated)\n" if len(lines) > 200 else "\n"))
        except Exception as e:
            print(f"[WARN] Could not write debug raw output: {e}", file=sys.stderr)

    # Split back to sentences and align tags
    sent_lens = [len(toks) for toks in sents_tokens]
    tagged_sents = split_flat_by_lengths(triples, sent_lens)
    xpos_seqs, upos_seqs = align_tags_to_input(sents_tokens, tagged_sents)

    # ---- Build output CSV in same schema ----
    if fmt == "sentence":
        xpos_joined, upos_joined = [], []
        for toks, xseq, useq in zip(sents_tokens, xpos_seqs, upos_seqs):
            n = len(toks)
            xseq = (xseq[:n] + ["_"] * max(0, n - len(xseq)))
            useq = (useq[:n] + ["_"] * max(0, n - len(useq)))
            xpos_joined.append(" ".join(xseq))
            upos_joined.append(" ".join(useq))
        df_out = df.copy()
        df_out["xpos_tags"] = xpos_joined
        if not args.no_upos:
            df_out["upos_tags"] = upos_joined

    else:  # token
        recs = []
        i_sent = 0
        use_upos = not args.no_upos
        for sid, g in df.groupby("sentence_id", sort=True):
            g = g.sort_values("token_id")
            words = g["word"].astype(str).tolist()
            xseq = xpos_seqs[i_sent] if i_sent < len(xpos_seqs) else []
            useq = upos_seqs[i_sent] if i_sent < len(upos_seqs) else []
            for j in range(len(words)):
                row = g.iloc[j]
                rec = {
                    "sentence_id": row["sentence_id"],
                    "token_id": row["token_id"],
                    "word": row["word"],
                    "xpos": xseq[j] if j < len(xseq) else "_"
                }
                if use_upos:
                    rec["upos"] = useq[j] if j < len(useq) else "_"
                recs.append(rec)
            i_sent += 1
        cols = ["sentence_id", "token_id", "word", "xpos"] + ([] if args.no_upos else ["upos"])
        df_out = pd.DataFrame.from_records(recs, columns=cols)

    df_out.to_csv(args.out, index=False, encoding=args.encoding)

    # Small console summary
    got = sum(len(x) for x in xpos_seqs)
    print(f"[OK] Wrote predictions to {args.out} (format={fmt})")
    print(f"[INFO] TreeTagger args used: {used_args or '(none)'}")
    print(f"[INFO] Tokens in: {total_tokens} | Parsed lines: {len(triples)} | Joined XPOS: {got}")

if __name__ == "__main__":
    main()