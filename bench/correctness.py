"""Check nano-vllm outputs against an external HF model and a saved baseline.

    python -m bench.correctness --write out/qwen3_0_6b_golden.json
    python -m bench.correctness --golden out/qwen3_0_6b_golden.json

Task C supplies greedy_match, teacher_forced_nll, and compare. The model
loading, fixed inputs, command-line handling, and file IO are scaffolding.
"""

import argparse
import json
import os
import sys


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, Neil Armstrong",
    "The three primary colors are",
    "Water boils at a temperature of",
    "To make a cup of tea, first",
    "The largest planet in our solar system is",
    "Once upon a time, in a small village,",
]
MAX_TOKENS = 32
TEMPERATURE = 1e-3  # SamplingParams forbids zero; bf16 ties can still differ.


def greedy_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def teacher_forced_nll(model, prompt_ids: list[int], output_ids: list[int]) -> float:
    import torch

    ids = torch.tensor([prompt_ids + output_ids], device=model.device)
    logits = model(ids).logits[0].float()

    start = len(prompt_ids)
    pred = logits[start - 1 : start - 1 + len(output_ids)]
    logp = torch.log_softmax(pred, dim=-1)

    target = torch.tensor(output_ids, device=logp.device)
    rows = torch.arange(len(output_ids), device=logp.device)
    token_logp = logp[rows, target]
    return -token_logp.mean().item()


def compare(results: list[dict], golden: list[dict]) -> list[str]:
    if [r["prompt"] for r in results] != [g["prompt"] for g in golden]:
        return ["prompts differ from the golden file"]

    def mean(rows, key):
        return sum(row[key] for row in rows) / len(rows)

    failures = []

    match = mean(results, "greedy_match")
    golden_match = mean(golden, "greedy_match")
    if match < golden_match / 2:
        failures.append(
            f"mean greedy_match {match:.1f} < half of golden {golden_match:.1f}"
        )

    nll = mean(results, "nll")
    golden_nll = mean(golden, "nll")
    if nll > golden_nll + 0.3:
        failures.append(
            f"mean nll {nll:.3f} > golden {golden_nll:.3f} + 0.3"
        )

    return failures


def nano_outputs(model_path: str, prompt_ids: list[list[int]]) -> list[list[int]]:
    from nanovllm import LLM, SamplingParams

    # Leave GPU memory for the HF judge in the same process.
    llm = LLM(
        model_path, enforce_eager=False, max_model_len=4096,
        gpu_memory_utilization=0.5,
    )
    params = SamplingParams(
        temperature=TEMPERATURE, ignore_eos=True, max_tokens=MAX_TOKENS,
    )
    return [output["token_ids"] for output in llm.generate(prompt_ids, params, use_tqdm=False)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", help="write results as the baseline on known-good code")
    mode.add_argument("--golden", help="compare to this baseline; exit 1 on failure")
    args = parser.parse_args()
    model_path = os.path.expanduser(args.model)
    golden = None
    if args.golden:
        with open(args.golden, encoding="utf-8") as source:
            golden = json.load(source)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = [tokenizer.encode(text) for text in PROMPTS]
    nano = nano_outputs(model_path, prompt_ids)
    hf = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).cuda().eval()
    print(f"GPU={torch.cuda.get_device_name(0)} model={model_path}")

    results = []
    with torch.no_grad():
        for text, ids, output in zip(PROMPTS, prompt_ids, nano):
            reference = hf.generate(
                torch.tensor([ids], device=hf.device), do_sample=False,
                max_new_tokens=MAX_TOKENS, min_new_tokens=MAX_TOKENS,
            )[0, len(ids):].tolist()
            results.append({
                "prompt": text,
                "greedy_match": greedy_match(output, reference),
                "nll": teacher_forced_nll(hf, ids, output),
                "output": tokenizer.decode(output),
            })

    for result in results:
        print(
            f"match={result['greedy_match']:>2}/{MAX_TOKENS}  "
            f"nll={result['nll']:.3f}  {result['prompt']!r} -> {result['output']!r}"
        )

    if args.write:
        os.makedirs(os.path.dirname(args.write) or ".", exist_ok=True)
        with open(args.write, "w", encoding="utf-8") as destination:
            json.dump(results, destination, indent=2, ensure_ascii=False)
            destination.write("\n")
        print(f"wrote {args.write}")
        return

    failures = compare(results, golden)
    for message in failures:
        print("FAIL", message)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
