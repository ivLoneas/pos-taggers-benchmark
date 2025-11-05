#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Async TreeTagger POS benchmark — no predictions written.

- Auto-detects sentence vs token CSV; preserves whitespace tokenization.
- Streams ALL tokens through ONE TreeTagger process per run (repeated N times).
- Model-only timing = time inside the external TreeTagger process (communicate()).
- Robust: retries with "-token" (if missing) and then with no extra args if output looks short.
- Outputs timing JSON (mean ± stdev + numeric stats + per-run raws).
- DOES NOT write any CSV predictions.

Example:
  python treetagger_pos_benchmark_async.py \
    --in data.csv \
    --tt-cmd /opt/treetagger/bin/tree-tagger \
    --tt-params /opt/treetagger/lib/english.par \
    --tt-args "-quiet" \
    --time-file treetagger/time.json \
    --runs 10 --warmup 1 --concurrency 1
"""

import argparse
import sys
import os
import shlex
import subprocess
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

def build_sentences_from_sentence_csv(df: pd.DataFrame) -> List[List[str]]:
    texts = df["text"].fillna("").astype(str).tolist()
    return [t.split() if t else [] for t in texts]

def build_sentences_from_token_csv(df: pd.DataFrame) -> List[List[str]]:
    sents: List[List[str]] = []
    for _, g in df.groupby("sentence_id", sort=True):
        g = g.sort_values("token_id")
        sents.append(g["word"].astype(str).tolist())
    return sents

def sentences_to_treetagger_input(sents: List[List[str]]) -> str:
    # One token per line; a blank line after each sentence is harmless.
    lines = []
    for toks in sents:
        lines.extend(toks)
        lines.append("")  # sentence boundary (optional)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")

# ---------- TreeTagger I/O ----------

def run_treetagger(tt_cmd: str, tt_params: str, tt_args_str: str, inp: str, encoding="utf-8") -> Tuple[str, str, int, float]:
    """
    Run TreeTagger once; return (stdout, stderr, returncode, model_time_seconds).
    model_time_seconds measures only the external process time (communicate()).
    """
    cmd: List[str] = shlex.split(tt_cmd)
    if tt_args_str:
        cmd += shlex.split(tt_args_str)
    if tt_params:
        cmd.append(tt_params)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=encoding,
        )
    except FileNotFoundError:
        raise RuntimeError(f"TreeTagger command not found: {tt_cmd}")
    except Exception as e:
        raise RuntimeError(f"Failed to start TreeTagger: {e}")

    t0 = time.perf_counter()
    out, err = proc.communicate(inp)
    model_time = time.perf_counter() - t0
    return out, err, proc.returncode, model_time

def parse_treetagger_output_flat(output: str) -> List[Tuple[str, str, str]]:
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

def default_time_path(inp: str, out_hint: str) -> str:
    base_dir = "."
    if out_hint:
        base_dir = os.path.dirname(os.path.abspath(out_hint)) or "."
    elif inp:
        base_dir = os.path.dirname(os.path.abspath(inp)) or "."
    return os.path.join(base_dir, "time.json")

# ---------- One timed run with retries ----------

def treetagger_single_pass(
    tt_cmd: str,
    tt_params: str,
    tt_args: str,
    tt_input: str,
    total_tokens: int,
    encoding: str,
    save_debug_head_to: str = None,
    debug_head_limit: int = 200,
) -> Tuple[float, float, int, str]:
    """
    Execute a *single* tagging pass with robust retries.
    Returns:
      - model_time_seconds (process time)
      - eval_loop_time_seconds (wall time including parsing/validation)
      - parsed_lines (triples count)
      - args_used (final args string)
    Optionally writes the head of raw output for debugging.
    """
    t_eval0 = time.perf_counter()

    # Attempt 1: as-is
    out1, err1, rc1, m1 = run_treetagger(tt_cmd, tt_params, tt_args, tt_input, encoding=encoding)
    triples1 = parse_treetagger_output_flat(out1)
    need_retry = (rc1 != 0) or (len(triples1) < max(1, int(0.8 * total_tokens)))

    chosen_out, chosen_err, chosen_m, chosen_args, chosen_triples = out1, err1, m1, tt_args, triples1

    # Attempt 2: append -token (if not already present)
    if need_retry:
        retry_args2 = tt_args if ("-token" in (tt_args or "")) else (f"{tt_args} -token").strip()
        out2, err2, rc2, m2 = run_treetagger(tt_cmd, tt_params, retry_args2, tt_input, encoding=encoding)
        triples2 = parse_treetagger_output_flat(out2)
        ok2 = (rc2 == 0) and (len(triples2) >= len(chosen_triples))
        if ok2:
            chosen_out, chosen_err, chosen_m, chosen_args, chosen_triples = out2, err2, m2, retry_args2, triples2
            need_retry = len(chosen_triples) < max(1, int(0.8 * total_tokens))
        else:
            need_retry = need_retry  # unchanged

    # Attempt 3: strip extra args entirely
    if need_retry and tt_args:
        out3, err3, rc3, m3 = run_treetagger(tt_cmd, tt_params, "", tt_input, encoding=encoding)
        triples3 = parse_treetagger_output_flat(out3)
        ok3 = (rc3 == 0) and (len(triples3) >= len(chosen_triples))
        if ok3:
            chosen_out, chosen_err, chosen_m, chosen_args, chosen_triples = out3, err3, m3, "", triples3

    # Optionally save debug head (only for the final chosen output)
    if save_debug_head_to:
        try:
            with open(save_debug_head_to, "w", encoding=encoding) as f:
                lines = chosen_out.splitlines()
                head = "\n".join(lines[:debug_head_limit])
                f.write(head + ("\n... (truncated)\n" if len(lines) > debug_head_limit else "\n"))
        except Exception as e:
            print(f"[WARN] Could not write debug raw output: {e}", file=sys.stderr)

    eval_time = time.perf_counter() - t_eval0
    return float(chosen_m), float(eval_time), int(len(chosen_triples)), chosen_args

# ---------- Async driver ----------

async def timed_run(
    run_idx: int,
    tt_cmd: str,
    tt_params: str,
    tt_args: str,
    tt_input: str,
    total_tokens: int,
    encoding: str,
    debug_head_path: str = None,
) -> Dict[str, Any]:
    def _work():
        m_time, eval_time, parsed_lines, args_used = treetagger_single_pass(
            tt_cmd=tt_cmd,
            tt_params=tt_params,
            tt_args=tt_args,
            tt_input=tt_input,
            total_tokens=total_tokens,
            encoding=encoding,
            save_debug_head_to=debug_head_path,  # only set for first run
        )
        return {
            "run": run_idx,
            "model_time_seconds": m_time,
            "evaluation_loop_time_seconds": eval_time,
            "parsed_lines": parsed_lines,
            "args_used": args_used,
            "tokens": int(total_tokens),
            "model_throughput_tokens_per_sec": float(safe_div(total_tokens, m_time)) if m_time > 0 else None,
            "overall_throughput_tokens_per_sec": float(safe_div(total_tokens, eval_time)) if eval_time > 0 else None,
        }
    return await asyncio.to_thread(_work)

async def benchmark_async(
    runs: int,
    tt_cmd: str,
    tt_params: str,
    tt_args: str,
    tt_input: str,
    total_tokens: int,
    encoding: str,
    concurrency: int = 1,
    warmup: int = 1,
    debug_head_path: str = None,
) -> Dict[str, Any]:
    # Warm-ups (excluded)
    for i in range(max(0, warmup)):
        await timed_run(-1 - i, tt_cmd, tt_params, tt_args, tt_input, total_tokens, encoding, None)

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def guarded(run_idx: int, dbg: str = None):
        async with sem:
            return await timed_run(run_idx, tt_cmd, tt_params, tt_args, tt_input, total_tokens, encoding, dbg)

    tasks = []
    for i in range(runs):
        # Write debug raw head only for the FIRST timed run, if requested
        dbg = debug_head_path if (i == 0 and debug_head_path) else None
        tasks.append(asyncio.create_task(guarded(i + 1, dbg)))
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["run"])
    return {"runs": results}

# ---------- Assemble timing JSON ----------

def assemble_timing_payload(
    fmt: str,
    tt_cmd: str,
    tt_params: str,
    tt_args: str,
    batch_hint: int,
    sentences: int,
    tokens: int,
    bench: Dict[str, Any],
) -> Dict[str, Any]:
    runs = bench["runs"]
    model_times = [r["model_time_seconds"] for r in runs]
    eval_times  = [r["evaluation_loop_time_seconds"] for r in runs]
    model_tps   = [r["model_throughput_tokens_per_sec"] for r in runs if r["model_throughput_tokens_per_sec"]]
    overall_tps = [r["overall_throughput_tokens_per_sec"] for r in runs if r["overall_throughput_tokens_per_sec"]]
    parsed_lines = [r.get("parsed_lines", 0) for r in runs]
    args_used_all = list({r.get("args_used", "") for r in runs})

    m_mean, m_std = mean_stdev(model_times)
    e_mean, e_std = mean_stdev(eval_times)
    mtps_mean, mtps_std = mean_stdev(model_tps) if model_tps else (None, None)
    otps_mean, otps_std = mean_stdev(overall_tps) if overall_tps else (None, None)

    return {
        "format": fmt,
        "framework": "treetagger",
        "tt_cmd": tt_cmd,
        "tt_params": tt_params,
        "tt_args_requested": tt_args,
        "tt_args_effective": args_used_all,  # unique set across runs
        "batch_hint": batch_hint,            # not used (single-shot), here for parity
        "sentences": int(sentences),
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

        # Sanity signals
        "parsed_lines_mean": round(float(stats.mean(parsed_lines)), 1) if parsed_lines else None,
        "parsed_lines_min": min(parsed_lines) if parsed_lines else None,
        "parsed_lines_max": max(parsed_lines) if parsed_lines else None,

        # Metadata
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_csv": None,  # benchmark-only
        "runs": runs,
    }

# ---------- Main ----------

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV (gold format)")
    ap.add_argument("--format", choices=["auto", "sentence", "token"], default="auto")

    ap.add_argument("--tt-cmd", required=True, help="TreeTagger command/path (e.g., '/opt/treetagger/bin/tree-tagger' or 'tree-tagger-english')")
    ap.add_argument("--tt-params", default=None, help="Parameter file path (e.g., '/opt/treetagger/lib/english.par'). Omit for language wrappers.")
    ap.add_argument("--tt-args", default="", help="Extra TreeTagger args (e.g., '-quiet'). Use single-dash flags.")
    ap.add_argument("--encoding", default="utf-8", help="File/stdio encoding")

    # Only to choose default placement for time.json if --time-file not given
    ap.add_argument("--out", dest="out_hint", default=None,
                    help="(Optional) Path hint to place time.json next to; no CSV will be written")

    # Debug
    ap.add_argument("--debug-raw-head", default=None, help="Write first ~200 lines of raw TreeTagger output from the first timed run")

    # Benchmark controls
    ap.add_argument("--runs", type=int, default=10, help="Number of timed runs (default: 10)")
    ap.add_argument("--warmup", type=int, default=1, help="Warm-up runs excluded from stats (default: 1)")
    ap.add_argument("--concurrency", type=int, default=1, help="Max concurrent runs (default: 1). Keep at 1 for stable CPU usage.")
    ap.add_argument("--time-file", default=None, help="Where to write timing JSON (default: next to --out/--in)")

    args = ap.parse_args()

    df = pd.read_csv(args.inp, encoding=args.encoding)
    fmt = args.format if args.format != "auto" else detect_format(df)

    # Build token sequences
    if fmt == "sentence":
        sents_tokens = build_sentences_from_sentence_csv(df)
        sentence_count = len(sents_tokens)
    else:
        sents_tokens = build_sentences_from_token_csv(df)
        sentence_count = len(sents_tokens)

    total_tokens = sum(len(x) for x in sents_tokens)
    tt_input = sentences_to_treetagger_input(sents_tokens)

    # Benchmark
    bench = await benchmark_async(
        runs=max(1, int(args.runs)),
        tt_cmd=args.tt_cmd,
        tt_params=args.tt_params,
        tt_args=args.tt_args,
        tt_input=tt_input,
        total_tokens=int(total_tokens),
        encoding=args.encoding,
        concurrency=max(1, int(args.concurrency)),
        warmup=max(0, int(args.warmup)),
        debug_head_path=args.debug_raw_head,
    )

    timing_payload = assemble_timing_payload(
        fmt=fmt,
        tt_cmd=args.tt_cmd,
        tt_params=args.tt_params,
        tt_args=args.tt_args,
        batch_hint=-1,
        sentences=sentence_count,
        tokens=int(total_tokens),
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
