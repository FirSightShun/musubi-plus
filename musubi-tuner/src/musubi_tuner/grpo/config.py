"""GRPO configuration dataclasses with TOML loading."""
from __future__ import annotations

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RewardConfig:
    name: str
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        return cls(
            name=d["name"],
            weight=float(d.get("weight", 1.0)),
            params=dict(d.get("params", {})),
        )


@dataclass
class GRPOConfig:
    # Sampling
    group_size: int = 4
    num_inference_steps: int = 20
    width: int = 256
    height: int = 256
    frame_count: int = 1
    guidance_scale: float = 1.0
    discrete_flow_shift: float = 14.5

    # Loss
    kl_coeff: float = 0.01
    clip_eps: float = 0.0  # 0 means disabled

    # Rewards
    rewards: list[RewardConfig] = field(default_factory=list)

    # Misc
    architecture: str = "hv"

    @classmethod
    def from_toml(cls, path: str | Path) -> "GRPOConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)

        grpo = data.get("grpo", data)  # support both [grpo] table and top-level

        reward_list = [RewardConfig.from_dict(r) for r in grpo.get("reward", [])]

        return cls(
            group_size=int(grpo.get("group_size", 4)),
            num_inference_steps=int(grpo.get("num_inference_steps", 20)),
            width=int(grpo.get("width", 256)),
            height=int(grpo.get("height", 256)),
            frame_count=int(grpo.get("frame_count", 1)),
            guidance_scale=float(grpo.get("guidance_scale", 1.0)),
            discrete_flow_shift=float(grpo.get("discrete_flow_shift", 14.5)),
            kl_coeff=float(grpo.get("kl_coeff", 0.01)),
            clip_eps=float(grpo.get("clip_eps", 0.0)),
            rewards=reward_list,
            architecture=str(grpo.get("architecture", "hv")),
        )
