"""NFT configuration dataclasses with TOML loading."""
from __future__ import annotations

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path

from musubi_tuner.grpo.config import RewardConfig  # reuse — no duplication


@dataclass
class NFTConfig:
    """Configuration for DiffusionNFT online RL training.

    See: https://arxiv.org/abs/2509.16117 (NVIDIA, ICLR 2026)

    TOML section: ``[nft]`` / ``[[nft.reward]]`` (same pattern as GRPO).
    """

    # Architecture
    architecture: str = "qwen_image"

    # Sampling
    group_size: int = 16           # samples per prompt for advantage estimation
    num_inference_steps: int = 10
    width: int = 512
    height: int = 512
    frame_count: int = 1
    guidance_scale: float = 1.0
    discrete_flow_shift: float = 2.2

    # NFT loss hyper-parameters
    beta: float = 1.0              # implicit positive/negative interpolation strength
    kl_coeff: float = 0.0001       # KL divergence regularization weight
    adv_clip_max: float = 5.0      # advantage clipping bound before [0,1] mapping

    # Old policy maintenance
    old_policy_update_every: int = 1   # update old policy every N training steps
    old_policy_decay: float = 0.0      # EMA decay: 0 = full copy, 0.5 = half-life EMA

    # Rewards
    rewards: list[RewardConfig] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str | Path) -> "NFTConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)

        nft = data.get("nft", data)  # support both [nft] table and top-level

        reward_list = [RewardConfig.from_dict(r) for r in nft.get("reward", [])]

        return cls(
            architecture=str(nft.get("architecture", "qwen_image")),
            group_size=int(nft.get("group_size", 16)),
            num_inference_steps=int(nft.get("num_inference_steps", 10)),
            width=int(nft.get("width", 512)),
            height=int(nft.get("height", 512)),
            frame_count=int(nft.get("frame_count", 1)),
            guidance_scale=float(nft.get("guidance_scale", 1.0)),
            discrete_flow_shift=float(nft.get("discrete_flow_shift", 2.2)),
            beta=float(nft.get("beta", 1.0)),
            kl_coeff=float(nft.get("kl_coeff", 0.0001)),
            adv_clip_max=float(nft.get("adv_clip_max", 5.0)),
            old_policy_update_every=int(nft.get("old_policy_update_every", 1)),
            old_policy_decay=float(nft.get("old_policy_decay", 0.0)),
            rewards=reward_list,
        )
