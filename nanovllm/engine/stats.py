from dataclasses import dataclass, field


@dataclass
class StepRecord:
    """One engine step. All durations are in seconds; no torch dependency."""

    is_prefill: bool
    num_seqs: int
    num_tokens: int
    wall: float
    cpu_schedule: float
    cpu_prepare: float
    cpu_launch: float
    cpu_sample_wait: float
    cpu_postprocess: float
    gpu_forward: float
    gpu_sample: float
    seq_ids: list[int] = field(default_factory=list)
    finished: list[int] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)

    @property
    def gpu_busy(self) -> float:
        return (self.gpu_forward + self.gpu_sample) / self.wall


@dataclass
class RequestRecord:
    seq_id: int
    num_prompt_tokens: int
    num_output_tokens: int
    arrival_time: float
    token_times: list[float]
    num_preemptions: int

    @property
    def ttft(self) -> float:
        return self.token_times[0] - self.arrival_time

    @property
    def tbt(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]
