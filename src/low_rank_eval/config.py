from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    name: str = "Qwen/Qwen3-4B-Instruct-2507"
    revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    trust_remote_code: bool = False
    load_in_4bit: bool = True
    quant_type: Literal["nf4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attention_backend: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"
    gradient_checkpointing: bool = True
    reference_parameter_count: int = Field(4_022_468_608, gt=0)


class DatasetRef(StrictModel):
    name: str
    revision: str
    config: str
    split: str


class DataConfig(StrictModel):
    ifeval_train: DatasetRef
    numina_train: DatasetRef
    ifeval_eval: DatasetRef
    gsm8k_train: DatasetRef
    gsm8k_eval: DatasetRef
    processed_dir: Path = Path("data/processed")
    num_proc: int = 8
    preparation_limit_per_task: int | None = Field(None, gt=0)


class DecontaminationConfig(StrictModel):
    approximate_threshold: float = Field(0.90, ge=0, le=1)
    candidate_threshold: float = Field(0.75, ge=0, le=1)
    num_perm: int = Field(64, ge=16)
    char_ngram_size: int = Field(5, ge=2)
    remove_internal_duplicates: bool = True

    @model_validator(mode="after")
    def candidate_must_be_permissive(self) -> DecontaminationConfig:
        if self.candidate_threshold > self.approximate_threshold:
            raise ValueError("candidate_threshold must not exceed approximate_threshold")
        return self


class TrainingConfig(StrictModel):
    train_token_budget_per_task: int = Field(1_000_000, gt=0)
    calibration_token_budget_per_task: int = Field(50_000, gt=0)
    max_sequence_length: int = Field(2048, gt=32)
    effective_batch_size: int = Field(16, gt=0)
    per_device_train_batch_size: int = Field(1, gt=0)
    gradient_accumulation_steps: int | Literal["auto"] = "auto"
    learning_rate: float = Field(2e-4, gt=0)
    num_train_epochs: float = Field(1.0, gt=0)
    max_steps: int = -1
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    weight_decay: float = Field(0.0, ge=0)
    lr_scheduler_type: str = "cosine"
    optimizer: str = "paged_adamw_8bit"
    seed: int = 42
    packing: bool = True
    save_steps: int = Field(25, gt=0)
    logging_steps: int = Field(1, gt=0)
    reset_optimizer_between_tasks: bool = True
    resume: bool = True

    def accumulation_for_world_size(self, world_size: int) -> int:
        denominator = world_size * self.per_device_train_batch_size
        if self.gradient_accumulation_steps != "auto":
            accumulation = self.gradient_accumulation_steps
            actual = denominator * accumulation
            if actual != self.effective_batch_size:
                raise ValueError(
                    f"effective_batch_size={self.effective_batch_size}, but "
                    f"world_size={world_size}, "
                    f"per_device={self.per_device_train_batch_size}, accumulation={accumulation} "
                    f"produce {actual}"
                )
            return accumulation
        if self.effective_batch_size % denominator:
            raise ValueError(
                "effective_batch_size must be divisible by world_size * per_device_train_batch_size"
            )
        return self.effective_batch_size // denominator


class LoraConfig(StrictModel):
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    strategy: Literal[
        "uniform", "early_heavy", "middle_heavy", "late_heavy", "random", "spectral"
    ] = "uniform"
    uniform_rank: int = Field(16, gt=0)
    low_rank: int = Field(8, gt=0)
    high_rank: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0, lt=1)
    budget_tolerance: float = Field(0.01, ge=0, lt=1)
    random_seed: int = 42
    rank_file: Path | None = None

    @model_validator(mode="after")
    def validate_targets(self) -> LoraConfig:
        allowed = {"q_proj", "k_proj", "v_proj", "o_proj"}
        if not self.target_modules or not set(self.target_modules) <= allowed:
            raise ValueError(f"target_modules must be a non-empty subset of {sorted(allowed)}")
        if self.strategy == "spectral" and self.rank_file is None:
            raise ValueError("spectral strategy requires rank_file")
        return self


class EvaluationConfig(StrictModel):
    engine: Literal["local_lm_eval_verifiers", "lm_eval"] = "local_lm_eval_verifiers"
    batch_size: int = Field(1, gt=0)
    calibration_batch_size: int = Field(1, gt=0)
    ifeval_max_new_tokens: int = Field(1280, gt=0)
    gsm8k_max_new_tokens: int = Field(512, gt=0)
    limit: int | None = Field(None, gt=0)
    temperature: float = 0.0
    do_sample: bool = False

    @model_validator(mode="after")
    def deterministic(self) -> EvaluationConfig:
        if self.do_sample or self.temperature != 0:
            raise ValueError("Scientific evaluation must use do_sample=false and temperature=0")
        return self


class SpectralConfig(StrictModel):
    pilot_rank: int = Field(32, gt=0)
    r_min: int = Field(4, gt=0)
    r_max: int = Field(32, gt=0)
    rank_step: int = Field(4, gt=0)
    baseline_rank: int = Field(16, gt=0)
    require_non_uniform: bool = True

    @model_validator(mode="after")
    def valid_grid(self) -> SpectralConfig:
        if self.r_min > self.baseline_rank or self.baseline_rank > self.r_max:
            raise ValueError("Expected r_min <= baseline_rank <= r_max")
        for value in (self.r_min, self.r_max, self.baseline_rank):
            if value % self.rank_step:
                raise ValueError("Spectral ranks must be multiples of rank_step")
        return self


class OutputConfig(StrictModel):
    results_dir: Path = Path("results")
    checkpoints_dir: Path = Path("checkpoints")
    run_name: str | None = None


class ExperimentConfig(StrictModel):
    schema_version: int = 1
    model: ModelConfig
    data: DataConfig
    decontamination: DecontaminationConfig = Field(default_factory=DecontaminationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    utility_lambda: float = Field(1.0, ge=0)

    def canonical_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json(exclude_none=False))

    def fingerprint(self) -> str:
        fingerprint_data = self.canonical_dict()
        # Calibration batching is an operational memory-control setting. It does not
        # change examples, model weights, deterministic generations, or the
        # token-weighted NLL definition. Excluding it preserves resumability when the
        # same experiment is evaluated with a safer batch size.
        fingerprint_data["evaluation"].pop("calibration_batch_size", None)
        payload = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def data_fingerprint(self) -> str:
        payload = {
            "clean_data": self.decontamination_fingerprint(),
            "tokenization": {
                "train_token_budget_per_task": self.training.train_token_budget_per_task,
                "calibration_token_budget_per_task": (
                    self.training.calibration_token_budget_per_task
                ),
                "max_sequence_length": self.training.max_sequence_length,
                "packing": self.training.packing,
                "seed": self.training.seed,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def decontamination_fingerprint(self) -> str:
        payload = {
            "model": {"name": self.model.name, "revision": self.model.revision},
            "data": self.data.model_dump(mode="json"),
            "decontamination": self.decontamination.model_dump(mode="json"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Cyclic config inheritance involving {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    return _deep_merge(_read_with_extends(parent_path, seen), raw)


def load_config(path: str | Path) -> ExperimentConfig:
    load_dotenv()
    return ExperimentConfig.model_validate(_read_with_extends(Path(path)))


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.canonical_dict(), handle, sort_keys=False)
