import logging
from typing import Optional

from PIL import Image

from src.utils.providers import get_provider

logger = logging.getLogger(__name__)


def stylize_for_objaverse(
    image_path: str,
    output_path: str,
    provider: str = "gemini",
    model_name: Optional[str] = None,
) -> Image.Image:
    """
    Apply Objaverse-style transfer to an image using the specified provider.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the stylized image.
        provider: Name of the style transfer provider (e.g., "gemini").
        model_name: Override the provider's default model. None uses the provider default.

    Returns:
        The stylized PIL Image.
    """
    provider_mod = get_provider(provider)
    image = Image.open(image_path).convert("RGB")

    kwargs = {}
    if model_name is not None:
        kwargs["model_name"] = model_name

    result = provider_mod.stylize_for_objaverse(image, **kwargs)
    result.save(output_path)
    logger.info(f"Stylized image saved to {output_path}")
    return result
