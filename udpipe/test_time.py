#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Async UDPipe benchmark (no predictions written).

- Runs the UDPipe tagging pass multiple times (default 10) and records timing metrics.
- Outputs JSON with "mean ± stdev" strings *and* numeric fields (mean, stdev, min, max).
- No CSV predictions are produced.

Example:
  pip install ufal.udpipe pandas
  python udpipe_benchmark_async.py \
    --in data.csv \
    --model udpipe/english-ewt-ud-2.5-191206.udpipe \
    --time-file udpipe/time.json \
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

# ---------- UDPipe helpers ----------

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

def run_pipeline_horizontal(model, sents, batch_size: int = 3000, parse: bool = False) -> Tuple[float, int]:
    """
    Process pretokenized sentences using UDPipe Pipeline with horizontal input.
    Returns:
      - model_time_seconds (sum of time spent inside pipe.process),
      - token_count (tokens seen by the model).
    """
    from ufal.udpipe import Pipeline, ProcessingError

    parser_stage = Pipeline.DEFAULT if parse else Pipeline.NONE
    pipe = Pipeline(model, "horizontal", Pipeline.DEFAULT, parser_stage, "conllu")

    model_time_sec = 0.0
    token_count = 0

    for i in range(0, len(sents), batch_size):
        block = sents[i:i + batch_size]
        text = "\n".join(" ".join(toks) for toks in block) + "\n"
        token_count += sum(len(toks) for toks in block)

        err = ProcessingError()
        t0 = time.perf_counter()
        _ = pipe.process(text, err)  # model-only timing
        model_time_sec += (time.perf_counter() - t0)

        if err.occurred():
            raise RuntimeError(f"UDPipe pipeline error: {err.message}")

    return model_time_sec, token_count

# ---------- Stats helpers ----------

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

# ---------- Async benchmark ----------

async def timed_tag_run(
    run_idx: int,
    model,
    sents: List[List[str]],
    batch_size: int,
    parse: bool,
) -> Dict[str, Any]:
    """
    Execute a single tagging pass in a worker thread and return timing metrics.
    """
    def _work():
        t0 = time.perf_counter()
        model_time_sec, token_count = run_pipeline_horizontal(
            model, sents, batch_size=batch_size, parse=parse
        )
        eval_loop_time_sec = time.perf_counter() - t0
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
    model,
    sents: List[List[str]],
    batch_size: int,
    parse: bool,
    concurrency: int = 1,
    warmup: int = 1,
) -> Dict[str, Any]:
    # Warm-up (excluded from reporting)
    for i in range(max(0, warmup)):
        await timed_tag_run(-1 - i, model, sents, batch_size, parse)

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def guarded(run_idx: int):
        async with sem:
            return await timed_tag_run(run_idx, model, sents, batch_size, parse)

    tasks = [asyncio.create_task(guarded(i + 1)) for i in range(runs)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["run"])
    return {"runs": results}

def assemble_timing_payload(
    fmt: str,
    model_path: str,
    batch_size: int,
    parse: bool,
    sents: List[List[str]],
    bench: Dict[str, Any],
    output_csv: str = None
) -> Dict[str, Any]:
    runs = bench["runs"]
    tokens = runs[0]["tokens"] if runs else sum(len(s) for s in sents)

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
        "model_path": os.path.abspath(model_path),
        "batch_size": int(batch_size),
        "parse": bool(parse),
        "sentences": int(len(sents)),
        "tokens": int(tokens),

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
        "output_csv": output_csv,  # always None here (no predictions)
        "runs": runs,
    }
    return payload

def default_time_path(inp: str, out: str) -> str:
    """
    Match the old 'next to --out' behavior when possible; otherwise next to --in; else CWD.
    """
    if out:
        base_dir = os.path.dirname(os.path.abspath(out)) or "."
    elif inp:
        base_dir = os.path.dirname(os.path.abspath(inp)) or "."
    else:
        base_dir = "."
    return os.path.join(base_dir, "time.json")

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--model", required=True, help="Path to the UDPipe .udpipe model (e.g., english-ewt-ud-2.5-191206.udpipe)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto")
    ap.add_argument("--batch-size", type=int, default=3000, help="Sentences per batch (horizontal input)")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding")
    ap.add_argument("--parse", action="store_true", help="Also run dependency parsing (not needed for POS)")

    # Legacy compat (only used to decide default time.json location if --time-file not given)
    ap.add_argument("--out", dest="out", default=None, help="(Optional) Used only to place time.json next to this path; no predictions are written")

    # Benchmark controls
    ap.add_argument("--runs", type=int, default=10, help="Number of timed runs (default: 10)")
    ap.add_argument("--warmup", type=int, default=1, help="Warm-up runs excluded from stats (default: 1)")
    ap.add_argument("--concurrency", type=int, default=1, help="Max concurrent timed runs (default: 1). Increase cautiously.")
    ap.add_argument("--time-file", default=None, help="Where to write timing JSON (default: time.json next to --out, else next to --in)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)
    sents = sentences_from_sentence_csv(df) if fmt == "sentence" else sentences_from_token_csv(df)

    model = ensure_model(args.model)

    # -------- Benchmark (N async runs) --------
    bench = await benchmark_async(
        runs=max(1, int(args.runs)),
        model=model,
        sents=sents,
        batch_size=args.batch_size,
        parse=args.parse,
        concurrency=max(1, int(args.concurrency)),
        warmup=max(0, int(args.warmup)),
    )

    timing_payload = assemble_timing_payload(
        fmt=fmt,
        model_path=args.model,
        batch_size=args.batch_size,
        parse=args.parse,
        sents=sents,
        bench=bench,
        output_csv=None,  # no predictions
    )

    # Determine time file path
    time_json_path = args.time_file or default_time_path(args.inp, args.out)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(time_json_path)), exist_ok=True)
        with open(time_json_path, "w", encoding="utf-8") as f:
            json.dump(timing_payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write timing JSON to {time_json_path}: {e}", file=sys.stderr)

    print(f"[OK] Benchmark timing written to {time_json_path}")
    m = timing_payload["model_time_seconds"]
    e = timing_payload["evaluation_loop_time_seconds"]
    print(f"[STATS] model_time_seconds: {m}")
    print(f"[STATS] evaluation_loop_time_seconds: {e}")

if __name__ == "__main__":
    asyncio.run(amain())
