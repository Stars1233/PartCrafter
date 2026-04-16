import io
import json
import logging
import os
import re

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = "gemini-3-flash-preview"
DEFAULT_STYLE_MODEL = "gemini-3.1-flash-image-preview"


def _get_client():
    """Initialize the google-genai client using an API key from the environment."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. "
            "Set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package is required for the Gemini provider. "
            "Install it with: pip install google-genai"
        )
    return genai.Client(api_key=api_key)


def _build_num_parts_prompt(max_num_parts: int, mode: str = "object") -> str:
    """Build the VLM prompt for suggesting the number of parts."""
    if mode == "object":
        return (
            "You are analyzing an image of a 3D object to determine how many "
            "structural parts it should be decomposed into for 3D mesh generation.\n\n"
            "Each part will become a separate 3D mesh. Think about how a 3D modeler "
            "would segment this object into physically separable, structurally distinct "
            "components.\n\n"
            "Guidelines:\n"
            "- Each part should be a single contiguous solid volume (no disconnected "
            "floating pieces within one part).\n"
            "- Symmetric components should be counted individually (e.g., a chair has "
            "4 separate legs, not 1 group of legs).\n"
            "- Parts should be non-overlapping and together cover the entire object.\n"
            f"- The number must be between 1 and {max_num_parts}.\n"
            "- Fewer large parts is better than many tiny parts -- only split where "
            "there is a clear structural boundary.\n\n"
            "Respond ONLY with a single integer representing the number of parts, "
            "no other text."
        )
    else:  # scene mode
        return (
            "You are analyzing an image of a 3D indoor scene to determine how many "
            "distinct objects are present for 3D mesh generation.\n\n"
            "Each object in the scene will become a separate 3D mesh. Count the "
            "individual furniture pieces, fixtures, and objects visible in the scene.\n\n"
            "Guidelines:\n"
            "- Each item should be a single distinct object (e.g., a table, a lamp, "
            "a plant).\n"
            "- Identical items should be counted individually (e.g., 3 chairs = 3, "
            "not 1).\n"
            "- Only count clearly visible, substantial objects -- ignore walls, floors, "
            "and ambient elements.\n"
            f"- The number must be between 1 and {max_num_parts}.\n\n"
            "Respond ONLY with a single integer representing the number of objects, "
            "no other text."
        )


STYLE_TRANSFER_PROMPT = (
    "Preserve all details of this image and perform style transfer to convert it into "
    "the visual style of a 3D rendering from the Objaverse dataset.\n\n"
    "The output should have these characteristics:\n"
    "- Clean, smooth surfaces with simplified textures\n"
    "- Soft, even lighting (as if rendered in a 3D engine with basic studio lighting)\n"
    "- White or very light neutral background\n"
    "- No real-world imperfections (noise, motion blur, lens distortion, reflections "
    "of the environment)\n"
    "- Colors and materials should look like a 3D asset preview render, not a photograph\n"
    "- Maintain the exact shape, proportions, pose, and all structural details of the "
    "original object\n\n"
    "Do NOT change the object itself, its geometry, or add/remove any elements. "
    "Only change the rendering style to look like a clean 3D model preview."
)


def _parse_int_response(text: str) -> int:
    """Extract an integer from a VLM response."""
    text = text.strip()
    # Try direct int parse first
    try:
        return int(text)
    except ValueError:
        pass
    # Look for first integer in the text
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract an integer from VLM response: {text!r}")


def suggest_num_parts(
    image: Image.Image,
    max_num_parts: int,
    mode: str = "object",
    model_name: str = DEFAULT_VLM_MODEL,
) -> int:
    """
    Use Gemini VLM to suggest how many parts an image should be decomposed into.

    Returns the suggested number of parts (int).
    """
    client = _get_client()
    prompt = _build_num_parts_prompt(max_num_parts, mode=mode)

    response = client.models.generate_content(
        model=model_name,
        contents=[image, prompt],
    )

    try:
        n = _parse_int_response(response.text)
    except ValueError:
        # Retry once
        logger.warning("First VLM response was not a valid integer, retrying...")
        response = client.models.generate_content(
            model=model_name,
            contents=[image, prompt],
        )
        n = _parse_int_response(response.text)

    return max(1, min(n, max_num_parts))


def stylize_for_objaverse(
    image: Image.Image,
    model_name: str = DEFAULT_STYLE_MODEL,
) -> Image.Image:
    """
    Use Gemini image generation to convert an image to Objaverse-style rendering.

    Returns the stylized PIL Image.
    """
    from google import genai

    client = _get_client()

    response = client.models.generate_content(
        model=model_name,
        contents=[image, STYLE_TRANSFER_PROMPT],
        config=genai.types.GenerateContentConfig(
            response_modalities=["image", "text"],
        ),
    )

    # Extract image from response
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            mime = part.inline_data.mime_type
            if mime and mime.startswith("image/"):
                return Image.open(io.BytesIO(part.inline_data.data))

    raise RuntimeError(
        "Style transfer failed: no image returned in the Gemini response. "
        "The model may not have generated an image for this input."
    )
