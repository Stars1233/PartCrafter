import logging
from typing import Optional

from PIL import Image

from src.utils.providers import get_provider

logger = logging.getLogger(__name__)


def suggest_num_parts(
    image_path: str,
    max_num_parts: int,
    mode: str = "object",
    provider: str = "gemini",
    model_name: Optional[str] = None,
) -> int:
    """
    Use a VLM to analyze an image and suggest how many parts it should be
    decomposed into.

    Args:
        image_path: Path to the input image.
        max_num_parts: Hard upper bound on number of parts.
        mode: "object" for single-object decomposition, "scene" for multi-object scenes.
        provider: Name of the VLM provider (e.g., "gemini").
        model_name: Override the provider's default model. None uses the provider default.

    Returns:
        The suggested number of parts (clamped to [1, max_num_parts]).
    """
    provider_mod = get_provider(provider)
    image = Image.open(image_path).convert("RGB")

    kwargs = {"mode": mode}
    if model_name is not None:
        kwargs["model_name"] = model_name

    n = provider_mod.suggest_num_parts(image, max_num_parts, **kwargs)

    # Clamp to valid range
    n = max(1, min(int(n), max_num_parts))
    return n
