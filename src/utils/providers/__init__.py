import importlib

PROVIDERS = {
    "gemini": "src.utils.providers.gemini_provider",
}


def get_provider(name: str):
    """Lazy-import and return the provider module by name."""
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return importlib.import_module(PROVIDERS[name])
