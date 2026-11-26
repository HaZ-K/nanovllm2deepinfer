"""Run one fixed workload; task B supplies the per-step driver."""

import argparse
import json
import os
from pathlib import Path
from random import Random
from time import perf_counter

from bench.metrics import format_report, summarize_requests, summarize_steps
from bench.trace import TraceWriter
from bench.workloads import SCENARIOS


def drive(llm, requests, seed: int = 0, num_steps: int | None = None) -> float:
    import torch
    from nanovllm import SamplingParams

    rng = Random(seed)
    pending = sorted(requests, key=lambda r: r.arrival_step)
    prompts = [[rng.randint(0, 10000) for _ in range(r.prompt_len)] for r in pending]
    i = virtual_step = engine_steps = 0
    t0 = perf_counter()

    while i < len(pending) or not llm.is_finished():
        if num_steps is not None and engine_steps >= num_steps:
            break

        while i < len(pending) and pending[i].arrival_step <= virtual_step:
            request = pending[i]
            params = SamplingParams(
                temperature=request.temperature,
                ignore_eos=True,
                max_tokens=request.max_tokens,
            )
            llm.add_request(prompts[i], params)
            i += 1

        if not llm.is_finished():
            with torch.cuda.nvtx.range(f"step {engine_steps}"):
                llm.step()
            engine_steps += 1
        virtual_step += 1

    return perf_counter() - t0


def warmup(llm, requests, n: int = 64):
    """Cover workload shapes and batch=1 before starting the measurement."""
    from nanovllm import SamplingParams

    rng = Random(-1)
    prompts = [[rng.randint(0, 10000) for _ in range(r.prompt_len)] for r in requests[:n]]
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    llm.generate(prompts, params, use_tqdm=False)
    llm.generate(prompts[:1], params, use_tqdm=False)
    llm.step_records.clear()
    llm.request_records.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="steady_decode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", help="write per-step JSONL here")
    parser.add_argument("--json", help="write the summary here")
    parser.add_argument("--num-steps", type=int, help="stop after N engine steps")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    if args.num_steps is not None and args.num_steps < 0:
        parser.error("--num-steps must be nonnegative")

    import torch
    from nanovllm import LLM

    engine_kwargs = {
        name: getattr(args, name)
        for name in (
            "max_num_seqs", "gpu_memory_utilization", "max_num_batched_tokens",
            "max_model_len", "enforce_eager",
        )
        if getattr(args, name) is not None
    }
    model_path = os.path.expanduser(args.model)
    llm = LLM(model_path, **engine_kwargs)
    workload = SCENARIOS[args.scenario](args.seed)
    warmup(llm, workload)
    hardware = torch.cuda.get_device_name(0)

    writer = None
    try:
        if args.trace:
            Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
            writer = llm.trace = TraceWriter(args.trace)
        torch.cuda.profiler.start()
        try:
            duration = drive(llm, workload, seed=args.seed, num_steps=args.num_steps)
        finally:
            torch.cuda.profiler.stop()
    finally:
        llm.trace = None
        if writer is not None:
            writer.close()

    steps = summarize_steps(llm.step_records)
    requests = summarize_requests(llm.request_records, duration)
    print(f"GPU={hardware} model={model_path}")
    print(format_report(args.scenario, steps, requests))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "scenario": args.scenario, "seed": args.seed,
            "hardware": hardware, "model": model_path,
            "dtype": str(llm.model_runner.config.hf_config.dtype),
            "duration_seconds": duration, "num_steps_limit": args.num_steps,
            "workload_requests": len(workload),
            "workload_completed": len(llm.request_records) == len(workload),
            "steps": steps, "requests": requests,
        }
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
