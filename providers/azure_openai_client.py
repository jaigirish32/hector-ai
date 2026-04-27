"""
Azure OpenAI client — uses the Azure-hosted Responses API.

File support: the Responses API on Azure accepts the same `input_file`
content blocks as OpenAI direct. Files are uploaded via the Files API
with purpose='assistants' (Azure doesn't accept 'user_data' yet, but
'assistants'-purpose files DO work with Responses API references on
api-version 2025-03-01-preview and later).
"""
from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError as OpenAIAuthError,
    AzureOpenAI,
    RateLimitError as OpenAIRateLimitError,
)

from providers.base import (
    AuthenticationError,
    BaseProviderClient,
    ChatRequest,
    ChatResponse,
    NotConfiguredError,
    ProviderError,
    RateLimitError,
    calculate_cost_usd,
)
from settings_manager import SecretKey, SettingsManager


# Responses API requires this version or later. Same minimum that
# enabled the Responses endpoint plus the Files API integration we
# verified during Phase 1.
AZURE_API_VERSION = "2025-03-01-preview"


class AzureOpenAIClient(BaseProviderClient):
    """Client for Azure-hosted OpenAI models via Responses API."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return (
            self._settings.has_secret(SecretKey.AZURE_OPENAI_API_KEY)
            and self._settings.has_secret(SecretKey.AZURE_OPENAI_ENDPOINT)
        )

    def complete(self, request: ChatRequest) -> ChatResponse:
        if not self.is_configured():
            raise NotConfiguredError(
                "Azure OpenAI not fully configured. Go to Settings and add "
                "your API key and endpoint URL."
            )

        deployment_key = (
            f"{SecretKey.AZURE_OPENAI_DEPLOYMENT_PREFIX}{request.model.id}"
        )
        deployment_name = self._settings.get_secret(deployment_key)
        if not deployment_name:
            raise NotConfiguredError(
                f"No Azure deployment name set for {request.model.label}. "
                f"Go to Settings and fill in the deployment name."
            )

        api_key = self._settings.get_secret(SecretKey.AZURE_OPENAI_API_KEY)
        endpoint = self._settings.get_secret(SecretKey.AZURE_OPENAI_ENDPOINT)

        client = AzureOpenAI(
            api_key=api_key,
            api_version=AZURE_API_VERSION,
            azure_endpoint=endpoint,
        )

        api_model = request.model.api_model_name

        # Newer GPT-5 family and reasoning models reject custom temperature.
        ignores_temperature = (
            api_model.startswith("gpt-5")
            or api_model.startswith("o1")
            or api_model.startswith("o3")
        )

        input_items = self._build_input(request)

        create_kwargs: dict = {
            "model": deployment_name,  # Azure uses deployment, not api_model
            "input": input_items,
            "max_output_tokens": request.max_tokens,
        }
        if not ignores_temperature:
            create_kwargs["temperature"] = request.temperature
        if request.system_prompt:
            create_kwargs["instructions"] = request.system_prompt

        start = time.monotonic()
        try:
            response = client.responses.create(**create_kwargs)
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "Azure rejected the API key. Check it's valid in Azure Portal.",
                raw=str(exc),
            ) from exc
        except OpenAIRateLimitError as exc:
            raise RateLimitError(
                "Azure rate limit hit. Wait a moment and retry.",
                raw=str(exc),
            ) from exc
        except APIConnectionError as exc:
            raise ProviderError(
                "Could not reach Azure OpenAI — check your endpoint URL "
                "and internet connection.",
                raw=str(exc),
            ) from exc
        except APIError as exc:
            message = getattr(exc, "message", str(exc))
            if "deploymentnotfound" in str(exc).lower() or "deployment" in message.lower():
                raise ProviderError(
                    f"Azure says the deployment '{deployment_name}' doesn't exist. "
                    f"Check the name in Azure Portal → your resource → Deployments.",
                    raw=str(exc),
                ) from exc
            raise ProviderError(
                f"Azure OpenAI error: {message}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error calling Azure OpenAI: {exc}",
                raw=str(exc),
            ) from exc

        latency = time.monotonic() - start

        text = getattr(response, "output_text", None) or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens)

        return ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=f"{api_model} (deployment: {deployment_name})",
        )

    # ---------- Internal helpers ----------

    def _build_input(self, request: ChatRequest) -> list[dict]:
        """Convert a ChatRequest into Responses API input format.

        Same shape as the OpenAI direct client: zero or more `input_file`
        blocks (one per attached file, filtered to this provider) followed
        by an `input_text` block with the user's prompt.

        Files come BEFORE the text — same convention as Anthropic and Gemini.
        """
        content_blocks: list[dict] = []

        for ref in request.file_refs:
            if ref.provider != "azure_openai":
                continue
            content_blocks.append({
                "type": "input_file",
                "file_id": ref.remote_id,
            })

        content_blocks.append({
            "type": "input_text",
            "text": request.prompt,
        })

        return [{
            "role": "user",
            "content": content_blocks,
        }]