#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run spaCy POS tagging over a CSV and save predictions in the same data format.

Supported input formats (auto-detected):

A) Sentence-level (one row per sentence)
    columns: [row_id?], text, [xpos_tags?], [upos_tags?]
    -> produces: xpos_tags (pred) and upos_tags (pred)

B) Token-level (one row per token)
    columns: sentence_id, token_id, word, [xpos?], [upos?]
    -> produces: sentence_id, token_id, word, xpos (pred) and upos (pred)

Notes:
- By default we PRESERVE your whitespace tokenization to keep alignment with gold.
  (We build spaCy Docs from your tokens with identical boundaries.)
- spaCy fields:
      token.tag_ -> XPOS (Penn Treebank-style, e.g., JJ, NN, VBZ)
      token.pos_ -> UPOS (Universal POS, e.g., ADJ, NOUN, VERB)
- If you prefer spaCy's own tokenizer (may break alignment), pass --use-spacy-tokenizer.

Usage:
    python predict_spacy_pos.py --in gold.csv --out pred.csv
    # token-level input:
    python predict_spacy_pos.py --in gold_tok.csv --out pred_tok.csv --format token
    # specify model:
    python predict_spacy_pos.py --in gold.csv --out pred.csv --model en_core_web_sm
"""

import argparse
import sys
from typing import List, Tuple
import pandas as pd

def detect_format(df: pd.DataFrame) -> str:
    cols = {c.lower() for c in df.columns}
    if {"sentence_id", "token_id", "word"} <= cols:
        return "token"
    if "text" in cols:
        return "sentence"
    return "sentence"

def _ensure_spacy(model: str = "en_core_web_sm", model_path: str = None):
    import spacy
    if model_path:
        try:
            return spacy.load(model_path)
        except Exception as e:
            print(f"[ERROR] Could not load model from --model-path={model_path}: {e}", file=sys.stderr)
            sys.exit(2)
    # try named model, try to download if missing
    try:
        return spacy.load(model)
    except OSError:
        print(f"[INFO] spaCy model '{model}' not found. Attempting to download...", file=sys.stderr)
        try:
            from spacy.cli import download
            download(model)
            return spacy.load(model)
        except Exception as e:
            print(f"[ERROR] Failed to download/load spaCy model '{model}'. "
                  f"Install it manually: python -m spacy download {model}\nReason: {e}", file=sys.stderr)
            sys.exit(2)

def _make_doc_from_tokens(nlp, tokens: List[str]):
    # Build a Doc with EXACT token boundaries (keeps your whitespace tokenization)
    from spacy.tokens import Doc
    spaces = [True] * (len(tokens) - 1) + [False] if tokens else []
    return Doc(nlp.vocab, words=tokens, spaces=spaces)

def _tag_tokens(doc) -> Tuple[List[str], List[str]]:
    # Return (xpos_tags, upos_tags) for a processed Doc
    xpos = [t.tag_ if t.tag_ else "_" for t in doc]
    upos = [t.pos_ if t.pos_ else "_" for t in doc]
    return xpos, upos

def run_sentence_mode(nlp, df: pd.DataFrame, use_spacy_tok: bool, batch_size: int) -> pd.DataFrame:
    out = df.copy()
    texts = out["text"].fillna("").astype(str).tolist()

    # Prepare docs, preserving tokenization unless told otherwise
    docs = []
    pre_tokens = []
    if use_spacy_tok:
        docs = list(nlp.pipe(texts, batch_size=batch_size))
    else:
        token_seqs = [t.split() if t else [] for t in texts]
        pre_tokens = token_seqs
        docs = list(nlp.pipe((_make_doc_from_tokens(nlp, toks) for toks in token_seqs),
                              batch_size=batch_size))

    xpos_pred, upos_pred = [], []
    for i, doc in enumerate(docs):
        x, u = _tag_tokens(doc)
        # Join back to space-separated sequences to match your evaluator
        xpos_pred.append(" ".join(x))
        upos_pred.append(" ".join(u))

    out["xpos_tags"] = xpos_pred
    out["upos_tags"] = upos_pred
    return out

def run_token_mode(nlp, df: pd.DataFrame, use_spacy_tok: bool, batch_size: int) -> pd.DataFrame:
    required = ["sentence_id", "token_id", "word"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Token format requires column '{col}'.")

    records = []
    # group per sentence, preserve order
    for sid, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        words = g["word"].astype(str).tolist()

        if use_spacy_tok:
            # Let spaCy re-tokenize from raw text; alignment with original tokens may change!
            text = " ".join(words)
            doc = nlp.make_doc(text)
            doc = nlp(doc)
        else:
            doc = _make_doc_from_tokens(nlp, words)
            doc = nlp(doc)

        xpos, upos = _tag_tokens(doc)
        # If spaCy re-tokenized, lengths may differ; we truncate to min for safety.
        m = min(len(words), len(xpos), len(upos))
        for (idx, row), tx, tu in zip(g.head(m).iterrows(), xpos[:m], upos[:m]):
            records.append({
                "sentence_id": row["sentence_id"],
                "token_id": row["token_id"],
                "word": row["word"],
                "xpos": tx,
                "upos": tu
            })
        # If lengths differ and you want strict checking, you could log it here.

    cols = ["sentence_id", "token_id", "word", "xpos", "upos"]
    return pd.DataFrame.from_records(records, columns=cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (predictions)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto",
                    help="Force input format (default: auto)")
    ap.add_argument("--model", default="en_core_web_sm", help="spaCy model name (e.g., en_core_web_sm)")
    ap.add_argument("--model-path", default=None, help="Path to a local spaCy model directory")
    ap.add_argument("--batch-size", type=int, default=1000, help="spaCy pipe batch size")
    ap.add_argument("--use-spacy-tokenizer", action="store_true",
                    help="Use spaCy's tokenizer instead of preserving your whitespace tokens (may break alignment).")
    args = ap.parse_args()

    try:
        import pandas as pd  # ensure pandas available even if imported above
        nlp = _ensure_spacy(model=args.model, model_path=args.model_path)
        df = pd.read_csv(args.inp)
        fmt = args.format if args.format != "auto" else detect_format(df)

        if fmt == "sentence":
            pred = run_sentence_mode(nlp, df, use_spacy_tok=args.use_spacy_tokenizer, batch_size=args.batch_size)
        else:
            pred = run_token_mode(nlp, df, use_spacy_tok=args.use_spacy_tokenizer, batch_size=args.batch_size)

        pred.to_csv(args.out, index=False)
        print(f"[OK] Wrote predictions to {args.out} (format={fmt}, model={args.model})")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()