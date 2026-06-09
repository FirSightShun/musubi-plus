from .base import BaseReward, register, build_rewards

# Import all built-in rewards to trigger registration
from . import hps  # noqa: F401
from . import pickscore  # noqa: F401
from . import image_reward  # noqa: F401
from . import clip  # noqa: F401
from . import ocr  # noqa: F401
from . import vlm  # noqa: F401
from . import delta_e  # noqa: F401

__all__ = ["BaseReward", "register", "build_rewards"]
