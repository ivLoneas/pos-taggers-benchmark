#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Async Stanza (StanfordNLP) POS benchmark — no predictions written.

- Runs the Stanza tagging pass multiple times (default 10) over pretokenized input.
- Records:
    * model_time_seconds (sum of nlp(batch) times)  -> "mean ± stdev"
    * evaluation_loop_time_seconds (wall time)      -> "mean ± stdev"
    * throughput (tokens/sec)                       -> "mean ± stdev"
- Writes a timing JSON (default: next to --out hint if given, else next to --in).
- Does NOT write any CSV predictions.

Example:
  pip install stanza pandas
  python stanza_pos_benchmark_async.py \
    --in data.csv \
    --lang en --batch-size 1000 --use-gpu --download \
    --time-file stanza/time.json \
    --runs 10 --warmup 1 --concurrency 1
"""

import argparse
from typing import List, Tuple, Dict, Any
import sys
import os
import json
import time
import asyncio
import statistics as stats
from datetime import datetime, timezone

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

def sentences_from_token_csv(df: pd.DataFrame) -> List[List[str]]:
    sents = []
    for _, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        sents.append(g["word"].astype(str).tolist())
    return sents

# -------- Stanza pipeline --------

def ensure_stanza_pipeline(lang: str, use_gpu: bool, model_dir: str, download_if_missing: bool):
    try:
        import stanza
    except Exception:
        print("[ERROR] stanza is not installed. Install with: pip install stanza", file=sys.stderr)
        raise

    model_dir_final = (
        model_dir
        or os.environ.get("STANZA_RESOURCES_DIR")
        or os.path.join(os.path.expanduser("~"), "stanza_resources")
    )
    os.makedirs(model_dir_final, exist_ok=True)

    def _build():
        return stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos",
            tokenize_pretokenized=True,  # we're passing pretokenized tokens
            use_gpu=use_gpu,
            dir=model_dir_final,
        )

    try:
        return _build()
    except Exception as e_first:
        if not download_if_missing:
            print(f"[ERROR] Could not build Stanza pipeline and --download not set: {e_first}", file=sys.stderr)
            raise
        try:
            stanza.download(lang, model_dir=model_dir_final, processors="tokenize,pos", verbose=False)
            return _build()
        except Exception as e:
            print(f"[ERROR] Failed to download or construct Stanza pipeline: {e}", file=sys.stderr)
            raise

# -------- Timed tagging (no predictions) --------

def stanza_tag_batch_time(nlp, batch_sents: List[List[str]]) -> Tuple[float, int]:
    """
    Tag one batch of pretokenized sentences and return:
      - elapsed model-only time (s) inside nlp(batch_sents)
      - number of tokens processed
    """
    if not batch_sents:
        return 0.0, 0
    num_tokens = sum(len(s) for s in batch_sents)
    t0 = time.perf_counter()
    doc = nlp(batch_sents)
    elapsed = time.perf_counter() - t0
    # Touch the doc to ensure full materialization
    _ = sum(len(s.words) for s in doc.sentences)
    return elapsed, num_tokens

def tag_all_pretokenized_benchmark(nlp, sents: List[List[str]], batch_size: int = 1000) -> Tuple[float, int]:
    """
    Run tagging over all sentences in batches; return:
      - total model_time_seconds (sum over batches)
      - total token_count
    """
    model_time_sec = 0.0
    token_count = 0
    for i in range(0, len(sents), batch_size):
        elapsed, ntoks = stanza_tag_batch_time(nlp, sents[i:i+batch_size])
        model_time_sec += elapsed
        token_count += ntoks
    return model_time_sec, token_count

# -------- Stats helpers --------

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

# -------- Async benchmark --------

async def timed_run(
    run_idx: int,
    nlp,
    sents: List[List[str]],
    batch_size: int,
) -> Dict[str, Any]:
    def _work():
        t_eval0 = time.perf_counter()
        model_time_sec, token_count = tag_all_pretokenized_benchmark(nlp, sents, batch_size=batch_size)
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
    nlp,
    sents: List[List[str]],
    batch_size: int,
    concurrency: int = 1,
    warmup: int = 1,
) -> Dict[str, Any]:
    for i in range(max(0, warmup)):
        await timed_run(-1 - i, nlp, sents, batch_size)

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def guarded(run_idx: int):
        async with sem:
            return await timed_run(run_idx, nlp, sents, batch_size)

    tasks = [asyncio.create_task(guarded(i + 1)) for i in range(runs)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["run"])
    return {"runs": results}

def assemble_timing_payload(
    fmt: str,
    lang: str,
    use_gpu: bool,
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
        "lang": lang,
        "use_gpu": bool(use_gpu),
        "batch_size": int(batch_size),
        "sentences": int(len(sents)),
        "tokens": int(tokens),

        # Pretty fields
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

# -------- Main CLI --------

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--format", choices=["auto","sentence","token"], default="auto", help="Force input format (default: auto)")
    ap.add_argument("--lang", default="en", help="Stanza language code (default: en)")
    ap.add_argument("--model-dir", default=None, help="Custom stanza model dir (e.g., /path/to/stanza_resources)")
    ap.add_argument("--batch-size", type=int, default=1000, help="Batch size (sentences) for Stanza processing")
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    ap.add_argument("--download", action="store_true", help="Download models if missing")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")

    # Only used to decide default location for time.json if --time-file is omitted
    ap.add_argument("--out", dest="out_hint", default=None,
                    help="(Optional) Just a path hint to place time.json next to; no CSV will be written")

    # Benchmark controls
    ap.add_argument("--runs", type=int, default=10, help="Number of timed runs (default: 10)")
    ap.add_argument("--warmup", type=int, default=1, help="Warm-up runs excluded from stats (default: 1)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Max concurrent timed runs (default: 1). Stanza is best benchmarked single-threaded.")
    ap.add_argument("--time-file", default=None, help="Where to write timing JSON (default: next to --out/--in)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)
    sents = sentences_from_sentence_csv(df) if fmt == "sentence" else sentences_from_token_csv(df)

    # Build pipeline (construction time is not part of model_time_seconds)
    nlp = ensure_stanza_pipeline(args.lang, args.use_gpu, args.model_dir, args.download)

    # Benchmark
    bench = await benchmark_async(
        runs=max(1, int(args.runs)),
        nlp=nlp,
        sents=sents,
        batch_size=args.batch_size,
        concurrency=max(1, int(args.concurrency)),
        warmup=max(0, int(args.warmup)),
    )

    timing_payload = assemble_timing_payload(
        fmt=fmt,
        lang=args.lang,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
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
