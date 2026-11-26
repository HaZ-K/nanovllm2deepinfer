"""Section 1.1 acceptance tests, added as each task is prepared.

The data-only module is loaded directly so CPU checks do not trigger
nanovllm.__init__ and its eager GPU-library imports.
"""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


STATS_PATH = Path(__file__).resolve().parents[1] / "nanovllm/engine/stats.py"
spec = importlib.util.spec_from_file_location("_section_1_1_stats", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)
StepRecord = stats.StepRecord
RequestRecord = stats.RequestRecord

CPU_FIELDS = [
    "cpu_schedule", "cpu_prepare", "cpu_launch",
    "cpu_sample_wait", "cpu_postprocess",
]
GPU_FIELDS = ["gpu_forward", "gpu_sample"]
MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def make_step(**overrides):
    values = dict(
        is_prefill=False, num_seqs=4, num_tokens=4, wall=0.010,
        cpu_schedule=0.001, cpu_prepare=0.001, cpu_launch=0.0005,
        cpu_sample_wait=0.006, cpu_postprocess=0.001,
        gpu_forward=0.006, gpu_sample=0.002,
    )
    values.update(overrides)
    return StepRecord(**values)


def test_gpu_busy():
    assert make_step().gpu_busy == pytest.approx(0.8)
    assert make_step(gpu_forward=0.0, gpu_sample=0.0).gpu_busy == 0.0


@pytest.fixture(scope="module")
def llm():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not os.path.isdir(MODEL):
        pytest.skip("needs a CUDA GPU and Qwen3-0.6B")
    from nanovllm import LLM
    return LLM(MODEL, enforce_eager=False, max_model_len=4096)


def test_step_records(llm):
    from nanovllm import SamplingParams

    llm.generate(["warmup"], SamplingParams(max_tokens=4), use_tqdm=False)
    llm.step_records.clear()
    traced = []
    llm.trace = traced.append
    try:
        params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16)
        llm.generate([[1] * 200, [2] * 300], params, use_tqdm=False)
    finally:
        llm.trace = None

    records = llm.step_records
    assert any(r.is_prefill for r in records)
    assert any(not r.is_prefill for r in records)
    assert len(traced) == len(records)
    assert all(actual is expected for actual, expected in zip(traced, records))
    for record in records:
        assert record.num_seqs > 0 and record.num_tokens > 0
        for name in CPU_FIELDS + GPU_FIELDS:
            assert getattr(record, name) >= 0, name
        assert record.gpu_forward > 0
        assert 0 < record.gpu_busy <= 1
        assert sum(getattr(record, name) for name in CPU_FIELDS) <= record.wall * 1.01
        # Detect the milliseconds-versus-seconds mistake.
        assert record.gpu_forward < record.wall


def _load_cpu_source(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sequence_module(monkeypatch):
    # Use the real data classes without importing the package's GPU entrypoint.
    sampling = _load_cpu_source("_section_1_1_sampling", "nanovllm/sampling_params.py")
    monkeypatch.setitem(sys.modules, "nanovllm.sampling_params", sampling)
    return _load_cpu_source("_section_1_1_sequence", "nanovllm/engine/sequence.py")


@pytest.fixture
def scheduler_module(monkeypatch, sequence_module):
    # KV kernels and model configuration are irrelevant to event bookkeeping.
    class Blocks:
        def __init__(self, *args):
            self.released = []

        def hash_blocks(self, seq):
            pass

        def deallocate(self, seq):
            self.released.append(seq)

    config = ModuleType("nanovllm.config")
    config.Config = object
    blocks = ModuleType("nanovllm.engine.block_manager")
    blocks.BlockManager = Blocks
    monkeypatch.setitem(sys.modules, "nanovllm.config", config)
    monkeypatch.setitem(sys.modules, "nanovllm.engine.sequence", sequence_module)
    monkeypatch.setitem(sys.modules, "nanovllm.engine.block_manager", blocks)
    return _load_cpu_source("_section_1_1_scheduler", "nanovllm/engine/scheduler.py")


def test_request_record():
    record = RequestRecord(0, 10, 3, 1.0, [1.5, 1.6, 1.8], 0)
    assert record.ttft == pytest.approx(0.5)
    assert record.tbt == pytest.approx([0.1, 0.2])


def test_sequence_timestamps(sequence_module, monkeypatch):
    clock = iter([1.0, 1.5, 1.6, 2.0])
    monkeypatch.setattr(sequence_module, "perf_counter", lambda: next(clock))
    seq = sequence_module.Sequence([10, 11])
    assert seq.arrival_time == 1.0
    assert seq.token_times == [] and seq.num_preemptions == 0
    seq.append_token(12)
    seq.append_token(13)
    assert seq.arrival_time == 1.0
    assert seq.token_times == [1.5, 1.6]
    assert len(seq.token_times) == seq.num_completion_tokens == 2

    other = sequence_module.Sequence([20])
    assert other.token_times == [] and other.token_times is not seq.token_times
    transport = seq.__getstate__()
    seq.arrival_time = 999.0
    seq.token_times.append(999.0)
    seq.num_preemptions = 99
    assert seq.__getstate__() == transport


def test_scheduler_events(scheduler_module, sequence_module):
    config = SimpleNamespace(
        max_num_seqs=4, max_num_batched_tokens=1024, eos=999,
        kvcache_block_size=256, num_kvcache_blocks=16,
    )
    scheduler = scheduler_module.Scheduler(config)
    assert scheduler.finished == [] and scheduler.preempted == []
    victim = sequence_module.Sequence([1, 2])
    arrival = victim.arrival_time
    scheduler.preempt(victim)
    assert victim.num_preemptions == 1 and victim.arrival_time == arrival
    assert scheduler.preempted == [victim.seq_id]
    assert scheduler.waiting[0] is victim

    sampling = sequence_module.SamplingParams
    short = sequence_module.Sequence([3, 4], sampling(max_tokens=1, ignore_eos=True))
    long = sequence_module.Sequence([5, 6], sampling(max_tokens=2, ignore_eos=True))
    for seq in (short, long):
        seq.status = sequence_module.SequenceStatus.RUNNING
        seq.num_scheduled_tokens = len(seq)
        scheduler.running.append(seq)
    scheduler.postprocess([short, long], [7, 8], is_prefill=True)
    assert scheduler.finished == [short]
    assert short.is_finished and not long.is_finished
    assert long in scheduler.running


@pytest.mark.parametrize("name", ["steady_decode", "mixed_long_short", "kv_pressure", "many_small"])
def test_workloads_deterministic(name):
    from bench.workloads import SCENARIOS
    make = SCENARIOS[name]
    assert make(seed=0) == make(seed=0)
    assert make(seed=0) != make(seed=1)


def test_mixed_long_short_shape():
    from bench.workloads import mixed_long_short
    requests = mixed_long_short(seed=0)
    longs = [request for request in requests if request.prompt_len == 3072]
    assert len(requests) - len(longs) == 128
    assert all(request.arrival_step == 0 for request in requests if request.prompt_len != 3072)
    steps = sorted(request.arrival_step for request in longs)
    assert steps[0] == 100
    assert all(b - a == 20 for a, b in zip(steps, steps[1:]))


def test_percentile():
    from bench.metrics import percentile
    assert percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)
    assert percentile([4, 1, 3, 2], 0) == 1
    assert percentile([4, 1, 3, 2], 100) == 4
    assert percentile([7], 99) == 7


def test_summarize():
    from bench.metrics import format_report, summarize_requests, summarize_steps
    steps = [
        make_step(is_prefill=True, wall=0.020, gpu_forward=0.018, gpu_sample=0.0),
        make_step(preempted=[3]),
        make_step(wall=0.030, gpu_forward=0.010, gpu_sample=0.0),
    ]
    summary = summarize_steps(steps)
    assert summary["prefill"]["steps"] == 1
    assert summary["decode"]["steps"] == 2
    assert summary["decode"]["gpu_busy"] == pytest.approx((0.008 + 0.010) / (0.010 + 0.030))
    assert summary["decode"]["wall"]["max"] == pytest.approx(0.030)
    assert summary["preemptions"] == 1
    requests = [
        RequestRecord(0, 10, 3, 0.0, [1.0, 1.1, 1.3], 0),
        RequestRecord(1, 10, 2, 0.0, [2.0, 2.4], 2),
    ]
    request_summary = summarize_requests(requests, duration=4.0)
    assert request_summary["throughput"] == pytest.approx(5 / 4.0)
    assert request_summary["ttft"]["max"] == pytest.approx(2.0)
    assert request_summary["tbt"]["max"] == pytest.approx(0.4)
    assert request_summary["preempted_requests"] == 1
    assert "gpu busy" in format_report("x", summary, request_summary)


def test_trace_roundtrip(tmp_path):
    from bench.trace import TraceWriter, read_trace
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer(make_step(seq_ids=[1, 2], finished=[2]))
    writer(make_step(is_prefill=True, preempted=[1]))
    writer.close()
    rows = read_trace(path)
    assert len(rows) == 2
    assert rows[0]["seq_ids"] == [1, 2] and rows[0]["finished"] == [2]
    assert rows[1]["is_prefill"] and rows[1]["preempted"] == [1]


def test_request_events(llm):
    from nanovllm import SamplingParams
    llm.step_records.clear()
    llm.request_records.clear()
    params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16)
    llm.generate([[1] * 200, [2] * 300], params, use_tqdm=False)
    requests = sorted(llm.request_records, key=lambda record: record.num_prompt_tokens)
    assert [record.num_prompt_tokens for record in requests] == [200, 300]
    for record in requests:
        assert record.num_output_tokens == len(record.token_times) == 16
        assert record.ttft > 0
        assert len(record.tbt) == 15 and all(gap > 0 for gap in record.tbt)
        assert record.num_preemptions == 0
    assert sum(len(step.finished) for step in llm.step_records) == 2
    assert {seq_id for step in llm.step_records for seq_id in step.finished} == {record.seq_id for record in requests}
    assert all(step.seq_ids for step in llm.step_records)
    assert llm.scheduler.finished == [] and llm.scheduler.preempted == []


def test_drive(llm):
    from bench.run import drive
    from bench.workloads import Request
    llm.step_records.clear()
    llm.request_records.clear()
    duration = drive(llm, [Request(200, 8, arrival_step=0), Request(300, 8, arrival_step=3)], seed=0)
    assert duration > 0 and len(llm.request_records) == 2
    late = next(record.seq_id for record in llm.request_records if record.num_prompt_tokens == 300)
    assert all(late not in step.seq_ids for step in llm.step_records[:3])

    # Virtual idle steps must not consume the actual engine-step budget.
    llm.step_records.clear()
    llm.request_records.clear()
    try:
        duration = drive(llm, [Request(200, 8, arrival_step=5)], seed=0, num_steps=2)
        assert duration > 0 and len(llm.step_records) == 2
        assert not llm.is_finished()
    finally:
        while not llm.is_finished():
            llm.step()


def test_greedy_match():
    from bench.correctness import greedy_match

    assert greedy_match([1, 2, 3, 4], [1, 9, 3, 4]) == 1
    assert greedy_match([1, 2], [1, 2, 3]) == 2
    assert greedy_match([7], [8]) == 0
    assert greedy_match([], [1]) == 0


def test_teacher_forced_nll():
    import math
    torch = pytest.importorskip("torch")
    from bench.correctness import teacher_forced_nll

    class Judge:
        device = "cpu"

        def __call__(self, ids):
            assert ids.tolist() == [[0, 1, 2, 3]]
            # Only rows 1 and 2 predict the two output tokens.
            probabilities = torch.tensor([
                [0.01, 0.01, 0.01, 0.97],
                [0.10, 0.10, 0.50, 0.30],
                [0.25, 0.25, 0.25, 0.25],
                [0.01, 0.01, 0.97, 0.01],
            ])
            return SimpleNamespace(logits=probabilities.log().unsqueeze(0))

    expected = (-math.log(0.5) - math.log(0.25)) / 2
    assert teacher_forced_nll(Judge(), [0, 1], [2, 3]) == pytest.approx(expected)


def test_compare():
    from bench.correctness import compare

    golden = [
        {"prompt": "a", "greedy_match": 32, "nll": 0.5},
        {"prompt": "b", "greedy_match": 32, "nll": 0.5},
    ]
    # One prompt can flip completely; acceptance uses the aggregate mean.
    boundary = [
        {"prompt": "a", "greedy_match": 0, "nll": 0.8},
        {"prompt": "b", "greedy_match": 32, "nll": 0.8},
    ]
    assert compare(golden, golden) == []
    assert compare(boundary, golden) == []
    bad_match = [dict(row, greedy_match=0) for row in golden]
    bad_nll = [dict(row, nll=0.81) for row in golden]
    assert len(compare(bad_match, golden)) == 1
    assert len(compare(bad_nll, golden)) == 1
    assert len(compare([dict(row, nll=0.81) for row in bad_match], golden)) == 2
    assert compare([dict(golden[0], prompt="other"), golden[1]], golden)
