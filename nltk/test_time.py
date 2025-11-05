#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Async NLTK POS benchmark — no predictions written.

- Builds a tokenized corpus from your CSV (sentence or token format).
- Tags with NLTK (PTB XPOS), and optionally also Universal POS (--with-upos).
- Repeats the tagging pass N times (default 10), async-dispatched via threads.
- Records mean ± stdev for model-only time and evaluation-loop wall time, plus throughput.
- Writes timing JSON next to --out hint (if provided) or next to --in (default).
- DOES NOT write any CSV predictions.

Example:
  pip install pandas nltk
  python nltk_pos_benchmark_async.py \
    --in data.csv \
    --with-upos \
    --time-file nltk/time.json \
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
import nltk

# ---------- NLTK resources ----------

def ensure_nltk_models():
    """
    Ensure NLTK tagger models are present.
    Prefers 'averaged_perceptron_tagger_eng' (new), falls back to 'averaged_perceptron_tagger' (old).
    """
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)

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

# ---------- Benchmark core ----------

def tag_batch_nltk(
    batch_sents: List[List[str]],
    with_upos: bool,
    lang: str,
) -> float:
    """
    Tag one batch and return *model-only* elapsed seconds.
    We time NLTK's tagger calls (PTB XPOS) and, if requested, the Universal pass too.
    """
    if not batch_sents:
        return 0.0

    # PTB XPOS pass
    t0 = time.perf_counter()
    xpos = nltk.pos_tag_sents(batch_sents, lang=lang)
    t1 = time.perf_counter()

    # Materialize to avoid lazy evaluation artifacts (should already be concrete)
    _ = sum(len(s) for s in xpos)

    elapsed = t1 - t0

    if with_upos:
        t2 = time.perf_counter()
        upos = nltk.pos_tag_sents(batch_sents, tagset="universal", lang=lang)
        t3 = time.perf_counter()
        _ = sum(len(s) for s in upos)
        elapsed += (t3 - t2)

    return elapsed

def tag_all_batched(
    all_sents: List[List[str]],
    with_upos: bool,
    lang: str,
    batch_size: int,
) -> Tuple[float, int]:
    """
    Tag the entire dataset in batches; return (model_time_seconds, token_count).
    """
    model_time_sec = 0.0
    token_count = sum(len(s) for s in all_sents)

    for i in range(0, len(all_sents), batch_size):
        batch = all_sents[i:i + batch_size]
        model_time_sec += tag_batch_nltk(batch, with_upos=with_upos, lang=lang)

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

# ---------- Async driver ----------

async def timed_run(
    run_idx: int,
    all_sents: List[List[str]],
    with_upos: bool,
    lang: str,
    batch_size: int,
) -> Dict[str, Any]:
    def _work():
        t_eval0 = time.perf_counter()
        model_time_sec, token_count = tag_all_batched(
            all_sents, with_upos=with_upos, lang=lang, batch_size=batch_size
        )
        eval_loop_time_sec = time.perf_counter() - t_eval0
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
    all_sents: List[List[str]],
    with_upos: bool,
    lang: str,
    batch_size: int,
    concurrency: int = 1,
    warmup: int = 1,
) -> Dict[str, Any]:
    for i in range(max(0, warmup)):
        await timed_run(-1 - i, all_sents, with_upos, lang, batch_size)

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def guarded(run_idx: int):
        async with sem:
            return await timed_run(run_idx, all_sents, with_upos, lang, batch_size)

    tasks = [asyncio.create_task(guarded(i + 1)) for i in range(runs)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["run"])
    return {"runs": results}

def assemble_timing_payload(
    fmt: str,
    lang: str,
    with_upos: bool,
    batch_size: int,
    sents: List[List[str]],
    bench: Dict[str, Any],
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

    return {
        "format": fmt,
        "framework": "nltk",
        "lang": lang,
        "with_upos": bool(with_upos),
        "batch_size": int(batch_size),
        "sentences": int(len(sents)),
        "tokens": int(tokens),

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

# ---------- Main ----------

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto",
                    help="Force input format (default: auto)")
    ap.add_argument("--with-upos", action="store_true", help="Also time a Universal POS pass (adds a second tagging pass)")
    ap.add_argument("--lang", default="eng", help="Language code for NLTK tagger (default: eng)")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding")
    ap.add_argument("--batch-size", type=int, default=1000, help="Sentences per batch for pos_tag_sents()")

    # Only used to choose default placement for time.json if --time-file not given
    ap.add_argument("--out", dest="out_hint", default=None,
                    help="(Optional) Path hint to place time.json next to; no CSV will be written")

    # Benchmark controls
    ap.add_argument("--runs", type=int, default=10, help="Number of timed runs (default: 10)")
    ap.add_argument("--warmup", type=int, default=1, help="Warm-up runs excluded from stats (default: 1)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Max concurrent timed runs (default: 1). NLTK tagger isn’t designed for high thread concurrency.")
    ap.add_argument("--time-file", default=None, help="Where to write timing JSON (default: next to --out/--in)")
    args = ap.parse_args()

    ensure_nltk_models()

    try:
        df = pd.read_csv(args.inp, encoding=args.encoding)
    except Exception as e:
        print(f"[ERROR] Could not read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format if args.format != "auto" else detect_format(df)
    sents = sentences_from_sentence_csv(df) if fmt == "sentence" else sentences_from_token_csv(df)

    # -------- Benchmark --------
    bench = await benchmark_async(
        runs=max(1, int(args.runs)),
        all_sents=sents,
        with_upos=bool(args.with_upos),
        lang=args.lang,
        batch_size=max(1, int(args.batch_size)),
        concurrency=max(1, int(args.concurrency)),
        warmup=max(0, int(args.warmup)),
    )

    timing_payload = assemble_timing_payload(
        fmt=fmt,
        lang=args.lang,
        with_upos=bool(args.with_upos),
        batch_size=int(args.batch_size),
        sents=sents,
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
