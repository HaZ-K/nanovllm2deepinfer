"""Capture short CPU/kernel timelines for learning to read a profiler.

    python -m bench.timeline --scenario steady_decode

Writes out/<scenario>_prefill.json (the initial window) and
out/<scenario>_decode.json (a later window), for https://ui.perfetto.dev.
With --small-prefill-tokens N, first writes a one-step _small_prefill.json after
warmup, before enqueueing the main workload. This is an added controlled
N-token probe, not a preempted request recomputing its KV cache. Per-window
labels and StepRecords go into _timeline_steps.json.
Every step's label comes from its actual StepRecord.is_prefill: a window can
contain both phases, especially when there are few prefill steps or preemptions.
All requests arrive together here; use bench.run for arrival_step experiments.
Profiler instrumentation adds CPU overhead, so inspect the shape of gaps rather
than treating these traces as benchmark timings.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from random import Random

from bench.run import warmup
from bench.workloads import SCENARIOS


def _label_steps(path: Path, labels: dict[str, str]):
    """Replace neutral capture scopes with the phases observed after each step."""
    trace = json.loads(path.read_text())
    for event in trace["traceEvents"]:
        name = event.get("name")
        if name in labels:
            event["name"] = labels[name]
    path.write_text(json.dumps(trace))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="steady_decode")
    parser.add_argument("--steps", type=int, default=3, help="engine steps per capture window")
    parser.add_argument("--small-prefill-tokens", type=int, default=0,
                        help="capture an isolated N-token probe before the main workload; 0 disables it")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.small_prefill_tokens < 0:
        parser.error("--small-prefill-tokens must be nonnegative")

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from nanovllm import LLM, SamplingParams

    llm = LLM(os.path.expanduser(args.model), enforce_eager=False, max_model_len=4096)
    requests = SCENARIOS[args.scenario](0)
    warmup(llm, requests)

    counters = {"prefill": 0, "decode": 0}
    windows = {}
    probe_matched = None

    def capture(window, num_steps=None):
        labels = {}
        records = []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for index in range(args.steps if num_steps is None else num_steps):
                if llm.is_finished():
                    break
                scope = f"engine step {index}"
                with record_function(scope):
                    llm.step()
                record = llm.step_records[-1]
                phase = "prefill" if record.is_prefill else "decode"
                labels[scope] = f"{phase} step {counters[phase]}"
                records.append({"label": labels[scope], **asdict(record)})
                counters[phase] += 1
            torch.cuda.synchronize()
        path = Path("out") / f"{args.scenario}_{window}.json"
        prof.export_chrome_trace(str(path))
        _label_steps(path, labels)
        windows[window] = {"trace": str(path), "steps": records}
        print(f"wrote {path}: {', '.join(labels.values())}")
        return records

    def save_steps():
        path = Path("out") / f"{args.scenario}_timeline_steps.json"
        path.write_text(json.dumps({
            "scenario": args.scenario, "model": os.path.expanduser(args.model),
            "small_prefill_tokens": args.small_prefill_tokens,
            "small_prefill_probe_matched": probe_matched, "windows": windows,
        }, indent=2) + "\n")
        print(f"wrote {path}")
        return path

    Path("out").mkdir(exist_ok=True)
    if args.small_prefill_tokens:
        size = args.small_prefill_tokens
        probe_rng = Random(-2)
        prompt = [probe_rng.randint(0, 10000) for _ in range(size)]
        llm.add_request(prompt, SamplingParams(max_tokens=1, ignore_eos=True))
        probe_id = llm.scheduler.waiting[-1].seq_id
        records = capture("small_prefill", num_steps=1)
        probe_matched = llm.is_finished() and len(records) == 1 and (
            records[0]["is_prefill"] and records[0]["num_seqs"] == 1
            and records[0]["num_tokens"] == size and records[0]["seq_ids"] == [probe_id]
        )
        if not probe_matched:
            path = save_steps()
            raise RuntimeError(
                f"the isolated probe did not finish in one {size}-token prefill step; "
                f"inspect {path} for the actual scheduled step"
            )

    rng = Random(0)
    for request in requests:
        prompt = [rng.randint(0, 10000) for _ in range(request.prompt_len)]
        llm.add_request(prompt, SamplingParams(
            temperature=request.temperature, ignore_eos=True,
            max_tokens=request.max_tokens,
        ))

    capture("prefill")
    while not llm.is_finished() and llm.step_records[-1].is_prefill:
        llm.step()
    for _ in range(20):
        if llm.is_finished():
            break
        llm.step()
    if not llm.is_finished():
        capture("decode")
    else:
        print("workload finished before the later capture window")
    save_steps()


if __name__ == "__main__":
    main()
