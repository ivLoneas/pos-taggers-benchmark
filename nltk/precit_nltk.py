#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run NLTK POS tagging over a CSV and save predictions in the same data format.

Supported input formats (auto-detected):

A) Sentence-level (one row per sentence)
    columns: [row_id?], text, xpos_tags?, upos_tags?
    -> produces: xpos_tags (pred) and optionally upos_tags (pred)

B) Token-level (one row per token)
    columns: sentence_id, token_id, word, xpos?, upos?
    -> produces: xpos (pred) and optionally upos (pred)

Notes:
- For sentence-level, tokens are taken from splitting `text` on whitespace.
  (This matches the CSV you built earlier where text is " ".join(tokens).)
- For token-level, tokens come from the `word` column grouped by sentence_id.
- XPOS uses the default NLTK PTB tagger (JJ, NN, VBZ, ...).
- UPOS uses NLTK's 'universal' tagset (ADJ, NOUN, VERB, ...). It's close to UD UPOS
  but not guaranteed identical to UD guidelines.

Usage:
    python predict_nltk_pos.py --in gold.csv --out pred.csv
    # force format and also output UPOS:
    python predict_nltk_pos.py --in gold_tok.csv --out pred_tok.csv --format token --with-upos
"""

import argparse
import sys
import pandas as pd
from typing import List, Tuple
import nltk

def ensure_nltk_models():
    """
    Try to ensure NLTK tagger resources are present.
    Newer NLTK uses 'averaged_perceptron_tagger_eng'; older uses 'averaged_perceptron_tagger'.
    """
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            pass
    # fallback
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)

    # universal tagset mapping needs no extra resource, but in case:
    # nothing to download; nltk maps PTB->universal internally.

def detect_format(df: pd.DataFrame) -> str:
    cols = set(c.lower() for c in df.columns)
    if {"sentence_id", "token_id", "word"} <= cols:
        return "token"
    if "text" in cols:
        return "sentence"
    # heuristic fallback
    return "sentence"

def pos_tag_tokens(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Return (xpos, upos) predictions for a list of tokens.
    """
    # XPOS (PTB)
    xpos_pairs = nltk.pos_tag(tokens, lang="eng")  # [(tok, 'NN'), ...]
    xpos = [t for _, t in xpos_pairs]
    # UPOS (universal)
    upos_pairs = nltk.pos_tag(tokens, tagset="universal", lang="eng")  # [(tok, 'NOUN'), ...]
    upos = [t for _, t in upos_pairs]
    return xpos, upos

def run_sentence_mode(df: pd.DataFrame, with_upos: bool) -> pd.DataFrame:
    # Copy to avoid mutating input
    out = df.copy()
    texts = out["text"].fillna("").astype(str).tolist()

    xpos_pred = []
    upos_pred = [] if with_upos else None

    for txt in texts:
        tokens = txt.split() if txt else []
        x, u = pos_tag_tokens(tokens)
        xpos_pred.append(" ".join(x))
        if with_upos:
            upos_pred.append(" ".join(u))

    out["xpos_tags"] = xpos_pred
    if with_upos:
        out["upos_tags"] = upos_pred
    return out

def run_token_mode(df: pd.DataFrame, with_upos: bool) -> pd.DataFrame:
    # We will produce a new DataFrame with the same required columns
    required = ["sentence_id", "token_id", "word"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Token format requires column '{col}'.")

    # group by sentence to preserve order, tag per sentence
    records = []
    for sid, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        tokens = g["word"].astype(str).tolist()
        x, u = pos_tag_tokens(tokens)
        # attach back to rows
        for (idx, row), xpos_tag, upos_tag in zip(g.iterrows(), x, u):
            rec = {
                "sentence_id": row["sentence_id"],
                "token_id": row["token_id"],
                "word": row["word"],
                "xpos": xpos_tag
            }
            if with_upos:
                rec["upos"] = upos_tag
            records.append(rec)

    cols = ["sentence_id", "token_id", "word", "xpos"] + (["upos"] if with_upos else [])
    out = pd.DataFrame.from_records(records, columns=cols)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (predictions)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto",
                    help="Force input format (default: auto)")
    ap.add_argument("--with-upos", action="store_true", help="Also output universal POS tags")
    args = ap.parse_args()

    ensure_nltk_models()

    try:
        df = pd.read_csv(args.inp)
    except Exception as e:
        print(f"[ERROR] Could not read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format if args.format != "auto" else detect_format(df)
    if fmt == "sentence":
        pred = run_sentence_mode(df, with_upos=args.with_upos)
    else:
        pred = run_token_mode(df, with_upos=args.with_upos)

    # Write predictions
    pred.to_csv(args.out, index=False)
    print(f"[OK] Wrote predictions to {args.out} (format={fmt}, with_upos={args.with_upos})")

if __name__ == "__main__":
    main()
