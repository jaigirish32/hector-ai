"""
Central registry of LLM models HECTOR-AI knows about.

Adding a new model is a single entry in DEFAULT_MODELS. Every other
part of the app (prompt area, response cards, settings, cost calculator)
reads from here rather than maintaining its own list.

NOTE ON AZURE OPENAI:
  Azure uses user-configured "deployment names" instead of global model
  names. The `api_model_name` field below holds the conceptual model
  identifier; the actual deployment name will be read from Settings
  at request time (one Azure resource URL + deployment map per install).
"""
from dataclasses import dataclass
from enum import Enum


class Provider(str, Enum):
    """Which company / runtime hosts a model."""

    OPENAI = "openai"              # api.openai.com
    AZURE_OPENAI = "azure_openai"  # user's Azure resource endpoint
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"                    # reserved for future (Grok)
    LOCAL = "local"                # reserved for future (Ollama, LM Studio)


@dataclass(frozen=True)
class ModelInfo:
    """Everything the app needs to know about one LLM model."""

    # Required
    id: str                 # Stable internal id, e.g. "gpt-4o-mini-azure"
    label: str              # Shown to user on chips and cards
    provider: Provider      # Routes the request to the right backend
    api_model_name: str     # What the provider's API expects

    # Optional — override per-model as needed
    supports_images: bool = False
    context_window: int = 128_000
    input_cost_per_1m: float = 0.0   # USD per 1M input tokens (approx)
    output_cost_per_1m: float = 0.0  # USD per 1M output tokens (approx)
    is_new: bool = False             # Shows a "NEW" badge in Settings
    enabled: bool = True             # Turn off without deleting


# ---------------------------------------------------------------------------
# The model registry. To add a new model, append one entry here.
# Costs below are approximate April-2026 rates — verify against each
# provider's current pricing page when billing matters.
# ---------------------------------------------------------------------------
DEFAULT_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="gpt-5.5",
        label="gpt-5.5",
        provider=Provider.OPENAI,
        api_model_name="gpt-5.5",
        supports_images=False,
        context_window=16_385,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
    ),
    ModelInfo(
        id="gpt-4.1-azure",
        label="gpt-4.1 Azure",
        provider=Provider.AZURE_OPENAI,
        api_model_name="gpt-4.1",
        supports_images=True,
        context_window=128_000,
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
    ),
    ModelInfo(
    id="gemini-2.5-flash",        # internal id stays for stability
    label="Gemini Flash (latest)", # honest label about behavior
    provider=Provider.GOOGLE,
    api_model_name="gemini-flash-latest",  # alias — always newest
    supports_images=True,
    context_window=1_000_000,
    input_cost_per_1m=0.30,        # approximate — may shift
    output_cost_per_1m=2.50,
    enabled=True,
    ),
    ModelInfo(
        id="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
        provider=Provider.ANTHROPIC,
        api_model_name="claude-sonnet-4-6",
        supports_images=True,
        context_window=200_000,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
    ),
]


# ---------------------------------------------------------------------------
# Lookup helpers — use these instead of iterating DEFAULT_MODELS yourself.
# ---------------------------------------------------------------------------
def get_model(model_id: str) -> ModelInfo | None:
    """Return the ModelInfo with this id, or None if not found."""
    for model in DEFAULT_MODELS:
        if model.id == model_id:
            return model
    return None


def enabled_models() -> list[ModelInfo]:
    """Return only the models currently enabled."""
    return [m for m in DEFAULT_MODELS if m.enabled]


def models_supporting_images() -> list[ModelInfo]:
    """Return only models that can process image inputs."""
    return [m for m in DEFAULT_MODELS if m.supports_images and m.enabled]