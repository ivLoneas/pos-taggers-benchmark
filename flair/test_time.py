#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Async Flair POS benchmark — no predictions written.

- Repeats Flair tagging N times (default 10) on sentence- or token-format CSV.
- Measures:
    * model_time_seconds: sum of time inside tagger.predict(...) calls
    * evaluation_loop_time_seconds: wall time around the tagging pass
    * throughput (tokens/sec)
- Preserves your whitespace tokenization by default (use --use-flair-tokenizer to let Flair retokenize).
- Writes timing JSON (mean ± stdev + numeric stats + raw runs).
- DOES NOT write any CSV predictions.

Example:
  pip install pandas flair
  python flair_pos_benchmark_async.py \
    --in data.csv \
    --xpos-model pos-fast --upos-model upos-fast \
    --time-file flair/time.json \
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
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd

# ----------------------------
# Utilities / I/O
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
    except Exception:
        print("[ERROR] Flair is not installed. Install with: pip install flair", file=sys.stderr)
        raise
    try:
        return SequenceTagger.load(model_name)
    except Exception as e:
        print(f"[ERROR] Could not load Flair model '{model_name}'. "
              f"Valid names include: 'pos', 'pos-fast', 'upos', 'upos-fast'.\nReason: {e}", file=sys.stderr)
        raise

def _make_sentence_from_tokens(tokens: List[str]):
    from flair.data import Sentence
    return Sentence(" ".join(tokens), use_tokenizer=False)  # preserve given tokens

def _make_sentence_from_text(text: str):
    from flair.data import Sentence
    return Sentence(text or "")  # Flair will tokenize

# ----------------------------
# Input preparation (no predictions)
# ----------------------------

def make_sentences_sentence(df: pd.DataFrame, use_flair_tok: bool):
    texts = df["text"].fillna("").astype(str).tolist()
    if use_flair_tok:
        def _make():
            return [_make_sentence_from_text(t) for t in texts]
    else:
        token_seqs = [t.split() if t else [] for t in texts]
        def _make():
            return [_make_sentence_from_tokens(ts) for ts in token_seqs]
    return _make, len(texts)

def make_sentences_token(df: pd.DataFrame, use_flair_tok: bool):
    groups = []
    for _, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        groups.append(g["word"].astype(str).tolist())
    if use_flair_tok:
        def _make():
            return [_make_sentence_from_text(" ".join(words)) for words in groups]
    else:
        def _make():
            return [_make_sentence_from_tokens(words) for words in groups]
    return _make, len(groups)

# ----------------------------
# Stats helpers
# ----------------------------

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

# ----------------------------
# Timed run (model-only vs wall)
# ----------------------------

def _predict_once(
    sentences,
    xpos_tagger,
    upos_tagger,
    batch_size: int,
) -> Tuple[float, int]:
    """
    Run Flair predict() for available taggers. Return (model_time_seconds, token_count).
    """
    model_time = 0.0

    if xpos_tagger is not None:
        t0 = time.perf_counter()
        xpos_tagger.predict(sentences, mini_batch_size=batch_size, verbose=False)
        model_time += (time.perf_counter() - t0)

    if upos_tagger is not None:
        t0 = time.perf_counter()
        upos_tagger.predict(sentences, mini_batch_size=batch_size, verbose=False)
        model_time += (time.perf_counter() - t0)

    # Count tokens seen by the model
    token_count = sum(len(s) for s in sentences)
    # Touch labels to avoid any lazy paths
    _ = 0
    for s in sentences:
        for tok in s:
            _ += len(tok.labels)

    return model_time, token_count

async def timed_run(
    run_idx: int,
    make_sentences,
    xpos_tagger,
    upos_tagger,
    batch_size: int,
) -> Dict[str, Any]:
    def _work():
        sentences = make_sentences()  # fresh Sentence objects each run
        t_eval0 = time.perf_counter()
        model_time, token_count = _predict_once(sentences, xpos_tagger, upos_tagger, batch_size)
        eval_loop_time = time.perf_counter() - t_eval0
        return {
            "run": run_idx,
            "model_time_seconds": float(model_time),
            "evaluation_loop_time_seconds": float(eval_loop_time),
            "model_throughput_tokens_per_sec": float(safe_div(token_count, model_time)) if model_time > 0 else None,
            "overall_throughput_tokens_per_sec": float(safe_div(token_count, eval_loop_time)) if eval_loop_time > 0 else None,
            "tokens": int(token_count),
        }
    return await asyncio.to_thread(_work)

async def benchmark_async(
    runs: int,
    make_sentences,
    xpos_tagger,
    upos_tagger,
    batch_size: int,
    concurrency: int = 1,
    warmup: int = 1,
) -> Dict[str, Any]:
    # Warm-ups (excluded)
    for i in range(max(0, warmup)):
        await timed_run(-1 - i, make_sentences, xpos_tagger, upos_tagger, batch_size)

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def guarded(run_idx: int):
        async with sem:
            return await timed_run(run_idx, make_sentences, xpos_tagger, upos_tagger, batch_size)

    tasks = [asyncio.create_task(guarded(i + 1)) for i in range(runs)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["run"])
    return {"runs": results}

# ----------------------------
# Assemble timing JSON
# ----------------------------

def assemble_timing_payload(
    fmt: str,
    models: Dict[str, Optional[str]],
    use_flair_tokenizer: bool,
    batch_size: int,
    sents_count: int,
    bench: Dict[str, Any],
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

    return {
        "format": fmt,
        "framework": "flair",
        "models": models,  # {"xpos": "...", "upos": "..."}
        "use_flair_tokenizer": bool(use_flair_tokenizer),
        "batch_size": int(batch_size),
        "sentences": int(sents_count),
        "tokens": int(tokens) if tokens is not None else None,

        # Pretty fields (mean ± stdev)
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
        "output_csv": None,  # benchmark-only
        "runs": runs,
    }

def default_time_path(inp: str, out_hint: str) -> str:
    base_dir = "."
    if out_hint:
        base_dir = os.path.dirname(os.path.abspath(out_hint)) or "."
    elif inp:
        base_dir = os.path.dirname(os.path.abspath(inp)) or "."
    return os.path.join(base_dir, "time.json")

# ----------------------------
# Main
# ----------------------------

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto",
                    help="Force input format (default: auto)")
    ap.add_argument("--xpos-model", default="pos-fast",
                    help="Flair model for XPOS (e.g., pos, pos-fast). Set empty to disable XPOS.")
    ap.add_argument("--upos-model", default="upos-fast",
                    help="Flair model for UPOS (e.g., upos, upos-fast). Set empty to disable UPOS.")
    ap.add_argument("--batch-size", type=int, default=32, help="Flair mini-batch size")
    ap.add_argument("--use-flair-tokenizer", action="store_true",
                    help="Use Flair's tokenizer (may change token boundaries).")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding")

    # Only for placing time.json if --time-file is omitted
    ap.add_argument("--out", dest="out_hint", default=None,
                    help="(Optional) Path hint to place time.json next to; no CSV will be written")

    # Benchmark controls
    ap.add_argument("--runs", type=int, default=10, help="Number of timed runs (default: 10)")
    ap.add_argument("--warmup", type=int, default=1, help="Warm-up runs excluded from stats (default: 1)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Max concurrent runs (default: 1). Flair/PyTorch models aren’t thread-safe for inference.")
    ap.add_argument("--time-file", default=None, help="Where to write timing JSON (default: next to --out/--in)")

    args = ap.parse_args()

    # Load CSV
    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)

    # Prepare sentence factory (creates fresh Sentence objects per run)
    if fmt == "sentence":
        make_sentences, sents_count = make_sentences_sentence(df, args.use_flair_tokenizer)
    else:
        make_sentences, sents_count = make_sentences_token(df, args.use_flair_tokenizer)

    # Load models (at least one required)
    xpos_tagger = None
    upos_tagger = None
    if args.xpos_model:
        xpos_tagger = _ensure_flair_tagger(args.xpos_model)
    if args.upos_model:
        upos_tagger = _ensure_flair_tagger(args.upos_model)
    if xpos_tagger is None and upos_tagger is None:
        print("[ERROR] Provide at least one of --xpos-model or --upos-model.", file=sys.stderr)
        sys.exit(2)

    # Benchmark
    bench = await benchmark_async(
        runs=max(1, int(args.runs)),
        make_sentences=make_sentences,
        xpos_tagger=xpos_tagger,
        upos_tagger=upos_tagger,
        batch_size=int(args.batch_size),
        concurrency=max(1, int(args.concurrency)),
        warmup=max(0, int(args.warmup)),
    )

    timing_payload = assemble_timing_payload(
        fmt=fmt,
        models={"xpos": args.xpos_model or None, "upos": args.upos_model or None},
        use_flair_tokenizer=bool(args.use_flair_tokenizer),
        batch_size=int(args.batch_size),
        sents_count=int(sents_count),
        bench=bench,
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
