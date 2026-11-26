"""Run bench.py's original workload and export the learner's step records.

    python -m scripts.inspect_step_timing

This is experiment/reporting scaffolding for task A. Timing is implemented
by the learner in ModelRunner and LLMEngine, not in this script.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
from random import randint, seed
from time import perf_counter


def main():
    import torch
    import transformers
    from nanovllm import LLM, SamplingParams

    seed(0)
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096)
    # Preserve bench.py's random-number consumption order: all prompts,
    # then all output lengths. The generated inputs are the same workload.
    prompts = [
        [randint(0, 10000) for _ in range(randint(100, 1024))]
        for _ in range(256)
    ]
    params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024))
        for _ in range(256)
    ]
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    llm.step_records.clear()
    start = perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    duration = perf_counter() - start

    records = [
        {"step": i, **asdict(record), "gpu_busy": record.gpu_busy}
        for i, record in enumerate(llm.step_records)
    ]
    total_tokens = sum(len(output["token_ids"]) for output in outputs)
    result = {
        "section": "1.1",
        "task": "A",
        "hardware": torch.cuda.get_device_name(0),
        "model": "Qwen/Qwen3-0.6B",
        "dtype": str(llm.model_runner.config.hf_config.dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": 0,
        "num_requests": 256,
        "duration_seconds": duration,
        "total_output_tokens": total_tokens,
        "output_tokens_per_second": total_tokens / duration,
        "record_durations_unit": "seconds",
        "wall_scope": "schedule start through postprocess completion",
        "gpu_busy_meaning": "GPU event span / wall; not kernel-active utilization",
        "records": records,
    }
    path = Path("out/1.1-task-a-records.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"GPU={result['hardware']} model={result['model']} dtype={result['dtype']}")
    print(f"Total: {total_tokens}tok, Time: {duration:.2f}s, Throughput: {total_tokens / duration:.2f}tok/s")
    for row in [row for row in records if not row["is_prefill"]][:5]:
        print(
            f"step={row['step']} bs={row['num_seqs']} "
            f"cpu_launch={row['cpu_launch'] * 1000:.2f}ms "
            f"cpu_sample_wait={row['cpu_sample_wait'] * 1000:.2f}ms "
            f"gpu_forward={row['gpu_forward'] * 1000:.2f}ms "
            f"gpu_sample={row['gpu_sample'] * 1000:.2f}ms "
            f"wall={row['wall'] * 1000:.2f}ms"
        )
    print(f"wrote {path} ({len(records)} steps)")


if __name__ == "__main__":
    main()
