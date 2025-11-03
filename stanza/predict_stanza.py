#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Stanza (Stanford NLP) POS tagging over a CSV and save predictions
in the same data format as your gold.

Formats (auto-detected):
A) Sentence-level CSV (one row per sentence)
   required: text
   optional: row_id, xpos_tags, upos_tags (ignored on input)
   -> outputs: xpos_tags, upos_tags  (space-separated sequences)

B) Token-level CSV (one row per token)
   required: sentence_id, token_id, word
   optional: xpos, upos (ignored on input)
   -> outputs: sentence_id, token_id, word, xpos, upos

Key points:
- Preserves your whitespace tokenization (tokenize_pretokenized=True).
- Processes data in batches to keep memory stable.
- Produces both XPOS (language-specific) and UPOS (universal).
- If models are missing and --download is given, downloads them.

Usage examples:
  python predict_stanza_pos.py --in gold_sent.csv --out pred_sent.csv
  python predict_stanza_pos.py --in gold_tok.csv  --out pred_tok.csv --format token
  python predict_stanza_pos.py --in gold.csv --out pred.csv --lang en --batch-size 200 --use-gpu --download
"""

import argparse
from typing import List, Tuple
import sys

import pandas as pd

# -------- CSV helpers --------

def detect_format(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if {"sentence_id", "token_id", "word"} <= cols:
        return "token"
    if "text" in cols:
        return "sentence"
    return "sentence"

def sentences_from_sentence_csv(df: pd.DataFrame) -> List[List[str]]:
    texts = df["text"].fillna("").astype(str).tolist()
    return [t.split() if t else [] for t in texts]

def sentences_from_token_csv(df: pd.DataFrame) -> Tuple[List[List[str]], List[Tuple[int,int]]]:
    """Return sentences (list of tokens) and an index map of (start_idx, end_idx) row slices (not used downstream but handy)."""
    sents = []
    bounds = []
    start = 0
    for sid, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        toks = g["word"].astype(str).tolist()
        sents.append(toks)
        end = start + len(toks)
        bounds.append((start, end))
        start = end
    return sents, bounds

def _pad(seq: List[str], n: int, fill: str = "_") -> List[str]:
    if len(seq) >= n:
        return seq[:n]
    return seq + [fill] * (n - len(seq))

# -------- Stanza runner --------

def ensure_stanza_pipeline(lang: str, use_gpu: bool, model_dir: str, download_if_missing: bool):
    """
    Build a Stanza pipeline, ensuring model_dir is a valid path.
    Falls back to ~/stanza_resources when model_dir is not provided.
    """
    import os
    try:
        import stanza
    except Exception as e:
        print("[ERROR] stanza is not installed. Install with: pip install stanza", file=sys.stderr)
        raise

    # Resolve a concrete model directory path
    model_dir_final = (
        model_dir
        or os.environ.get("STANZA_RESOURCES_DIR")
        or os.path.join(os.path.expanduser("~"), "stanza_resources")
    )
    os.makedirs(model_dir_final, exist_ok=True)

    # Try to build pipeline first (fast path)
    try:
        nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos",
            tokenize_pretokenized=True,
            use_gpu=use_gpu,
            dir=model_dir_final,   # NOTE: 'dir' is the correct kw for Pipeline
        )
        return nlp
    except Exception as e_first:
        if not download_if_missing:
            print(f"[ERROR] Could not build Stanza pipeline and --download not set: {e_first}", file=sys.stderr)
            raise

    # Download models into model_dir_final, then retry
    try:
        # Newer stanza uses model_dir kw; older used dir. This call works on current releases:
        stanza.download(lang, model_dir=model_dir_final, processors="tokenize,pos", verbose=False)
        nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos",
            tokenize_pretokenized=True,
            use_gpu=use_gpu,
            dir=model_dir_final,
        )
        return nlp
    except Exception as e:
        print(f"[ERROR] Failed to download or construct Stanza pipeline: {e}", file=sys.stderr)
        raise


def stanza_tag_batch(nlp, batch_sents: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Tag a batch of pre-tokenized sentences with Stanza.
    Returns two lists (same length as batch_sents):
      - xpos_seqs: list of XPOS tag sequences per sentence
      - upos_seqs: list of UPOS tag sequences per sentence
    """
    if not batch_sents:
        return [], []
    # With tokenize_pretokenized=True, you can pass list-of-lists directly
    doc = nlp(batch_sents)
    xpos_out, upos_out = [], []
    for sent in doc.sentences:
        # Prefer words (one per token in pretokenized mode)
        xpos = [(w.xpos or "_") for w in sent.words]
        upos = [(w.upos or "_") for w in sent.words]
        xpos_out.append(xpos)
        upos_out.append(upos)
    return xpos_out, upos_out

def tag_all_pretokenized(nlp, sents: List[List[str]], batch_size: int = 1000) -> Tuple[List[List[str]], List[List[str]]]:
    xpos_all, upos_all = [], []
    for i in range(0, len(sents), batch_size):
        xb, ub = stanza_tag_batch(nlp, sents[i:i+batch_size])
        xpos_all.extend(xb)
        upos_all.extend(ub)
    return xpos_all, upos_all

# -------- Main CLI --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (predictions)")
    ap.add_argument("--format", choices=["auto","sentence","token"], default="auto", help="Force input format (default: auto)")
    ap.add_argument("--lang", default="en", help="Stanza language code (default: en)")
    ap.add_argument("--model-dir", default=None, help="Custom stanza model dir (e.g., /path/to/stanza_resources)")
    ap.add_argument("--batch-size", type=int, default=1000, help="Batch size (sentences) for Stanza processing")
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    ap.add_argument("--download", action="store_true", help="Download models if missing")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)

    # Build pretokenized sentences
    if fmt == "sentence":
        sents = sentences_from_sentence_csv(df)
    else:
        sents, _ = sentences_from_token_csv(df)

    # Init Stanza
    nlp = ensure_stanza_pipeline(args.lang, args.use_gpu, args.model_dir, args.download)

    # Tag everything in batches
    xpos_seqs, upos_seqs = tag_all_pretokenized(nlp, sents, batch_size=args.batch_size)

    # Build outputs in matching schema
    if fmt == "sentence":
        # Join sequences (pad just in case)
        xpos_joined, upos_joined = [], []
        for toks, xseq, useq in zip(sents, xpos_seqs, upos_seqs):
            n = len(toks)
            xpos_joined.append(" ".join(_pad(xseq, n)))
            upos_joined.append(" ".join(_pad(useq, n)))
        out_df = df.copy()
        out_df["xpos_tags"] = xpos_joined
        out_df["upos_tags"] = upos_joined

    else:  # token-level
        recs = []
        i_sent = 0
        for sid, g in df.groupby("sentence_id", sort=True):
            g = g.sort_values("token_id")
            toks = g["word"].astype(str).tolist()
            xseq = xpos_seqs[i_sent] if i_sent < len(xpos_seqs) else []
            useq = upos_seqs[i_sent] if i_sent < len(upos_seqs) else []
            for j in range(len(toks)):
                row = g.iloc[j]
                recs.append({
                    "sentence_id": row["sentence_id"],
                    "token_id": row["token_id"],
                    "word": row["word"],
                    "xpos": xseq[j] if j < len(xseq) else "_",
                    "upos": useq[j] if j < len(useq) else "_",
                })
            i_sent += 1
        out_df = pd.DataFrame.from_records(recs, columns=["sentence_id","token_id","word","xpos","upos"])

    out_df.to_csv(args.out, index=False, encoding=args.encoding)
    print(f"[OK] Wrote predictions to {args.out} (format={fmt}, lang={args.lang}, batch_size={args.batch_size})")

if __name__ == "__main__":
    main()