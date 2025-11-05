#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Async spaCy POS benchmark (no predictions written).

- Runs the spaCy tagging pass multiple times (default 10) and records timing metrics.
- Outputs JSON with "mean ± stdev" strings AND numeric fields (mean, stdev, min, max, per-run raws).
- No CSV predictions are produced.

Example:
  pip install spacy pandas
  python spacy_pos_benchmark_async.py \
    --in data.csv \
    --model en_core_web_sm \
    --time-file spacy/time.json \
    --runs 10 --warmup 1 --concurrency 1
"""

import argparse
import sys
import os
import json
import time
import asyncio
import statistics as stats
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

import pandas as pd

# ---------- helpers ----------

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
    from spacy.tokens import Doc
    spaces = [True] * (len(tokens) - 1) + [False] if tokens else []
    return Doc(nlp.vocab, words=tokens, spaces=spaces)

# ---------- input prep (no predictions) ----------

def prepare_inputs_sentence(nlp, df: pd.DataFrame, use_spacy_tok: bool):
    texts = df["text"].fillna("").astype(str).tolist()
    if use_spacy_tok:
        # spaCy will tokenize from raw text each run
        def make_inputs():
            return texts
    else:
        # Pre-tokenized: build fresh Docs each run (outside measured model time)
        token_seqs = [t.split() if t else [] for t in texts]
        def make_inputs():
            return [_make_doc_from_tokens(nlp, toks) for toks in token_seqs]
    return make_inputs

def prepare_inputs_token(nlp, df: pd.DataFrame, use_spacy_tok: bool):
    # group into sentences
    groups = []
    for _, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        groups.append(g["word"].astype(str).tolist())
    if use_spacy_tok:
        def make_inputs():
            return [" ".join(words) for words in groups]
    else:
        def make_inputs():
            return [_make_doc_from_tokens(nlp, words) for words in groups]
    return make_inputs

# ---------- stats helpers ----------

def safe_div(n, d):
    return (n / d) if d and d > 0 else None

def mean_stdev(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (float(stats.mean(xs)), float(stats.stdev(xs)))

def fmt_pm(mean_val: float, stdev_val: float, precision: int = 6, unit: str = "s") -> str:
    return f"{round(mean_val, precision)} ± {round(stdev_val, precision)} {unit}"

# ---------- async timed run ----------

async def timed_tag_run(
    run_idx: int,
    nlp,
    make_inputs,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Execute a single tagging pass in a worker thread and return timing metrics.
    Measures:
      - model_time_seconds: time for nlp.pipe(...) enumeration only
      - evaluation_loop_time_seconds: wall time around the whole tagging pass
    """
    def _work():
        # Fresh inputs each run to avoid reusing annotated Docs
        inputs = make_inputs()
        t_eval0 = time.perf_counter()
        t_model0 = time.perf_counter()
        docs = list(nlp.pipe(inputs, batch_size=batch_size))
        model_time_sec = time.perf_counter() - t_model0
        eval_loop_time_sec = time.perf_counter() - t_eval0
        token_count = sum(len(d) for d in docs)
        return {
            "run": run_idx,
            "model_time_seconds": float(model_time_sec),
            "evaluation_loop_time_seconds": float(eval_loop_time_sec),
            "model_throughput_tokens_per_sec": float(safe_div(token_count, model_time_sec)) if model_time_sec > 0 else None,
            "overall_throughput_tokens_per_sec": float(safe_div(token_count, eval_loop_time_sec)) if eval_loop_time_sec > 0 else None,
            "tokens": int(token_count),
        }
    return await asyncio.to_thread(_work)

async def benchmark_async(
    runs: int,
    nlp,
    make_inputs,
    batch_size: int,
    concurrency: int = 1,
    warmup: int = 1,
) -> Dict[str, Any]:
    # Warm-ups (excluded)
    for i in range(max(0, warmup)):
        await timed_tag_run(-1 - i, nlp, make_inputs, batch_size)

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def guarded(run_idx: int):
        async with sem:
            return await timed_tag_run(run_idx, nlp, make_inputs, batch_size)

    tasks = [asyncio.create_task(guarded(i + 1)) for i in range(runs)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["run"])
    return {"runs": results}

def assemble_timing_payload(
    fmt: str,
    model_name: str,
    model_path: str,
    nlp,
    batch_size: int,
    use_spacy_tokenizer: bool,
    sents_count: int,
    bench: Dict[str, Any],
    out_hint: str = None
) -> Dict[str, Any]:
    runs = bench["runs"]
    tokens = runs[0]["tokens"] if runs else None

    model_times = [r["model_time_seconds"] for r in runs]
    eval_times  = [r["evaluation_loop_time_seconds"] for r in runs]
    model_tps   = [r["model_throughput_tokens_per_sec"] for r in runs if r["model_throughput_tokens_per_sec"]]
    overall_tps = [r["overall_throughput_tokens_per_sec"] for r in runs if r["overall_throughput_tokens_per_sec"]]

    m_mean, m_std = mean_stdev(model_times)
    e_mean, e_std = mean_stdev(eval_times)
    mtps_mean, mtps_std = mean_stdev(model_tps) if model_tps else (None, None)
    otps_mean, otps_std = mean_stdev(overall_tps) if overall_tps else (None, None)

    payload = {
        "format": fmt,
        "model": model_name,
        "model_path": os.path.abspath(model_path) if model_path else None,
        "pipe_components": list(getattr(nlp, "pipe_names", [])),
        "use_spacy_tokenizer": bool(use_spacy_tokenizer),
        "batch_size": int(batch_size),
        "sentences": int(sents_count),
        "tokens": int(tokens) if tokens is not None else None,

        # Preferred fields (pretty: mean ± stdev)
        "model_time_seconds": fmt_pm(m_mean, m_std, precision=6, unit="s"),
        "evaluation_loop_time_seconds": fmt_pm(e_mean, e_std, precision=6, unit="s"),
        "model_throughput_tokens_per_sec":
            (f"{round(mtps_mean,3)} ± {round(mtps_std,3)} tok/s" if mtps_mean is not None else None),
        "overall_throughput_tokens_per_sec":
            (f"{round(otps_mean,3)} ± {round(otps_std,3)} tok/s" if otps_mean is not None else None),

        # Numeric details
        "model_time_seconds_mean": round(m_mean, 6),
        "model_time_seconds_stdev": round(m_std, 6),
        "model_time_seconds_min": round(min(model_times), 6) if model_times else None,
        "model_time_seconds_max": round(max(model_times), 6) if model_times else None,

        "evaluation_loop_time_seconds_mean": round(e_mean, 6),
        "evaluation_loop_time_seconds_stdev": round(e_std, 6),
        "evaluation_loop_time_seconds_min": round(min(eval_times), 6) if eval_times else None,
        "evaluation_loop_time_seconds_max": round(max(eval_times), 6) if eval_times else None,

        "model_throughput_tokens_per_sec_mean": round(mtps_mean, 3) if mtps_mean is not None else None,
        "model_throughput_tokens_per_sec_stdev": round(mtps_std, 3) if mtps_std is not None else None,
        "overall_throughput_tokens_per_sec_mean": round(otps_mean, 3) if otps_mean is not None else None,
        "overall_throughput_tokens_per_sec_stdev": round(otps_std, 3) if otps_std is not None else None,

        # Metadata
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_csv": None,  # no predictions in this benchmark
        "runs": runs,
    }
    return payload

def default_time_path(inp: str, out_hint: str) -> str:
    # Match "next to --out" behavior if provided; else next to --in; else CWD
    base_dir = "."
    if out_hint:
        base_dir = os.path.dirname(os.path.abspath(out_hint)) or "."
    elif inp:
        base_dir = os.path.dirname(os.path.abspath(inp)) or "."
    return os.path.join(base_dir, "time.json")

# ---------- main ----------

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto",
                    help="Force input format (default: auto)")
    ap.add_argument("--model", default="en_core_web_sm", help="spaCy model name (e.g., en_core_web_sm)")
    ap.add_argument("--model-path", default=None, help="Path to a local spaCy model directory")
    ap.add_argument("--batch-size", type=int, default=1000, help="spaCy pipe batch size")
    ap.add_argument("--use-spacy-tokenizer", action="store_true",
                    help="Use spaCy's tokenizer instead of preserving whitespace tokens (may change tokenization).")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding")

    # Only used to choose default placement for time.json if --time-file not given
    ap.add_argument("--out", dest="out_hint", default=None,
                    help="(Optional) Used only to place time.json next to this path; no predictions are written")

    # Benchmark controls
    ap.add_argument("--runs", type=int, default=10, help="Number of timed runs (default: 10)")
    ap.add_argument("--warmup", type=int, default=1, help="Warm-up runs excluded from stats (default: 1)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Max concurrent timed runs (default: 1). spaCy Language is not guaranteed thread-safe.")
    ap.add_argument("--time-file", default=None, help="Where to write timing JSON (default: time.json next to --out/--in)")
    args = ap.parse_args()

    nlp = _ensure_spacy(model=args.model, model_path=args.model_path)
    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)

    if fmt == "sentence":
        make_inputs = prepare_inputs_sentence(nlp, df, use_spacy_tok=args.use_spacy_tokenizer)
        sents_count = len(df)
    else:
        make_inputs = prepare_inputs_token(nlp, df, use_spacy_tok=args.use_spacy_tokenizer)
        sents_count = df["sentence_id"].nunique()

    # -------- Benchmark (N async runs) --------
    bench = await benchmark_async(
        runs=max(1, int(args.runs)),
        nlp=nlp,
        make_inputs=make_inputs,
        batch_size=args.batch_size,
        concurrency=max(1, int(args.concurrency)),
        warmup=max(0, int(args.warmup)),
    )

    # -------- Assemble + write timing JSON --------
    timing_payload = assemble_timing_payload(
        fmt=fmt,
        model_name=args.model,
        model_path=args.model_path,
        nlp=nlp,
        batch_size=args.batch_size,
        use_spacy_tokenizer=args.use_spacy_tokenizer,
        sents_count=sents_count,
        bench=bench,
        out_hint=args.out_hint,
    )

    time_json_path = args.time_file or default_time_path(args.inp, args.out_hint)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(time_json_path)), exist_ok=True)
        with open(time_json_path, "w", encoding="utf-8") as f:
            json.dump(timing_payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write timing JSON to {time_json_path}: {e}", file=sys.stderr)

    print(f"[OK] Benchmark timing written to {time_json_path}")
    print(f"[STATS] model_time_seconds: {timing_payload['model_time_seconds']}")
    print(f"[STATS] evaluation_loop_time_seconds: {timing_payload['evaluation_loop_time_seconds']}")

if __name__ == "__main__":
    asyncio.run(amain())
