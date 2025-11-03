#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict POS tags using UDPipe 1 via the Python package `ufal.udpipe` (Pipeline API).

- Input CSV: auto-detects sentence vs token format.
- Preserves your whitespace tokenization (pre-tokenized).
- Uses Pipeline(model, "horizontal", "tag", "", "conllu") to get UPOS/XPOS.
- Outputs the same schema as your gold:
    * sentence CSV -> xpos_tags, upos_tags (space-separated)
    * token CSV    -> sentence_id, token_id, word, xpos, upos

Run:
  pip install ufal.udpipe pandas
  python predict_udpipe_pos.py \
    --in data.csv \
    --out udpipe/preds.csv \
    --model udpipe/english-ewt-ud-2.5-191206.udpipe \
    --debug-raw-out udpipe/raw_head.conllu
"""

import argparse
import sys
from typing import List, Tuple
import pandas as pd

# ---------- CSV helpers ----------

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

def sentences_from_token_csv(df: pd.DataFrame) -> List[List[str]]:
    sents = []
    for _, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        sents.append(g["word"].astype(str).tolist())
    return sents

def pad(seq: List[str], n: int, fill: str = "_") -> List[str]:
    return seq[:n] if len(seq) >= n else seq + [fill] * (n - len(seq))

# ---------- UDPipe helpers (ufal.udpipe) ----------

def ensure_model(model_path: str):
    try:
        from ufal.udpipe import Model
    except Exception:
        print("[ERROR] ufal.udpipe is not installed. Install with: pip install ufal.udpipe", file=sys.stderr)
        raise
    model = Model.load(model_path)
    if model is None:
        raise RuntimeError(f"Cannot load UDPipe model: {model_path}")
    return model

def make_horizontal_text(sents: List[List[str]]) -> str:
    """One sentence per line; tokens space-separated (pretokenized)."""
    return "\n".join(" ".join(toks) for toks in sents) + "\n"

def run_pipeline_horizontal(model, sents, batch_size: int = 3000, parse: bool = False) -> str:
    """
    Process pretokenized sentences using UDPipe Pipeline with horizontal input.
    Returns concatenated CoNLL-U. Runs tagging by default; parsing optional.
    """
    from ufal.udpipe import Pipeline, ProcessingError

    # Tag only by default; enable parse if requested
    parser_stage = Pipeline.DEFAULT if parse else Pipeline.NONE
    pipe = Pipeline(model, "horizontal", Pipeline.DEFAULT, parser_stage, "conllu")

    conllu_parts = []
    for i in range(0, len(sents), batch_size):
        block = sents[i:i + batch_size]
        text = "\n".join(" ".join(toks) for toks in block) + "\n"

        # NEW: create a fresh ProcessingError each time (no .clear() needed/available)
        err = ProcessingError()
        out = pipe.process(text, err)
        if err.occurred():
            raise RuntimeError(f"UDPipe pipeline error: {err.message}")

        conllu_parts.append(out)

    return "".join(conllu_parts)

# ---------- CoNLL-U parsing ----------

def conllu_flat_tags(conllu_text: str) -> List[Tuple[str, str, str]]:
    """
    Parse CoNLL-U and return a FLAT list of (FORM, UPOS, XPOS) per token.
    Skips comments and multiword/empty nodes (IDs like '1-2' or '1.1').
    """
    triples = []
    for line in conllu_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            cols = line.split()
            if len(cols) < 5:
                continue
        tok_id = cols[0]
        if "-" in tok_id or "." in tok_id:
            continue
        form = cols[1]
        upos = cols[3] if len(cols) > 3 else "_"
        xpos = cols[4] if len(cols) > 4 else "_"
        triples.append((form, upos or "_", xpos or "_"))
    return triples

def split_flat_by_lengths(flat_triples: List[Tuple[str, str, str]], sent_lens: List[int]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Split FLAT (FORM, UPOS, XPOS) back to sentences using original lengths.
    Returns (xpos_seqs, upos_seqs) aligned to input.
    """
    xpos_seqs, upos_seqs = [], []
    i = 0
    n = len(flat_triples)
    for L in sent_lens:
        j = min(i + L, n)
        chunk = flat_triples[i:j]
        xpos_seqs.append([x for (_, _, x) in chunk])
        upos_seqs.append([u for (_, u, _) in chunk])
        i = j
    return xpos_seqs, upos_seqs

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (predictions)")
    ap.add_argument("--model", required=True, help="Path to the UDPipe .udpipe model (e.g., english-ewt-ud-2.5-191206.udpipe)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto")
    ap.add_argument("--batch-size", type=int, default=3000, help="Sentences per batch (horizontal input)")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding")
    ap.add_argument("--debug-raw-out", default=None, help="Save first ~200 lines of tagged CoNLL-U here (optional)")
    ap.add_argument("--parse", action="store_true", help="Also run dependency parsing (not needed for POS)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)

    # Build pretokenized sentences from your CSV
    if fmt == "sentence":
        sents = sentences_from_sentence_csv(df)
    else:
        sents = sentences_from_token_csv(df)

    model = ensure_model(args.model)

    # Process with UDPipe Pipeline (horizontal, tag only by default)
    conllu = run_pipeline_horizontal(model, sents, batch_size=args.batch_size, parse=args.parse)

    # Optional: dump CoNLL-U head for debugging
    if args.debug_raw_out:
        try:
            with open(args.debug_raw_out, "w", encoding=args.encoding) as f:
                lines = conllu.splitlines()
                head = "\n".join(lines[:200])
                f.write(head + ("\n... (truncated)\n" if len(lines) > 200 else "\n"))
        except Exception as e:
            print(f"[WARN] Could not write debug output: {e}", file=sys.stderr)

    # Parse CoNLL-U and split back using known lengths
    flat = conllu_flat_tags(conllu)
    sent_lens = [len(t) for t in sents]
    xpos_seqs, upos_seqs = split_flat_by_lengths(flat, sent_lens)

    # Emit predictions in the same schema
    if fmt == "sentence":
        xpos_joined, upos_joined = [], []
        for toks, xseq, useq in zip(sents, xpos_seqs, upos_seqs):
            n = len(toks)
            xpos_joined.append(" ".join(pad(xseq, n)))
            upos_joined.append(" ".join(pad(useq, n)))
        out_df = df.copy()
        out_df["xpos_tags"] = xpos_joined
        out_df["upos_tags"] = upos_joined
    else:
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
    total_in = sum(len(s) for s in sents)
    total_out = sum(len(x) for x in xpos_seqs)
    print(f"[OK] Wrote predictions to {args.out} (format={fmt})")
    print(f"[INFO] Tokens in: {total_in} | Tagged tokens (xpos): {total_out}")

if __name__ == "__main__":
    main()