"""
Provider Manager providing LLM abstraction with automatic failover (OpenRouter -> Gemini).
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import requests

from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    name: str
    api_key: Optional[str]
    model: str
    base_url: str
    priority: int
    enabled: bool = True


class ProviderManager:
    """Manages LLM providers with automatic failover on 429/timeout/5xx errors."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers: List[ProviderConfig] = self._build_provider_chain(settings)
        self.active_index: int = 0

    def _build_provider_chain(self, s: Settings) -> List[ProviderConfig]:
        chain = []
        if s.openrouter_api_key:
            chain.append(ProviderConfig(
                name="OpenRouter",
                api_key=s.openrouter_api_key,
                model=s.llm_model,
                base_url="https://openrouter.ai/api/v1",
                priority=1,
            ))
        if s.gemini_api_key:
            chain.append(ProviderConfig(
                name="Google Gemini",
                api_key=s.gemini_api_key,
                model=s.gemini_model,
                base_url="https://generativelanguage.googleapis.com/v1beta",
                priority=2,
            ))
        return chain

    def get_active_provider_info(self) -> Tuple[str, str]:
        """Return (provider_name, model_name) of the active provider."""
        if not self.providers:
            return "No Provider", "N/A"
        active = self.providers[self.active_index]
        return active.name, active.model

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """Generate chat completion using the active provider with automatic failover."""
        if not self.providers:
            raise ValueError("No LLM API keys configured. Set OPENROUTER_API_KEY or GEMINI_API_KEY in .env.")

        last_exception = None

        # Try starting from current active provider index
        indices_to_try = list(range(self.active_index, len(self.providers))) + list(range(0, self.active_index))

        for idx in indices_to_try:
            provider = self.providers[idx]
            if not provider.enabled or not provider.api_key:
                continue

            try:
                if provider.name == "OpenRouter":
                    res = self._call_openrouter(provider, messages, temperature)
                elif provider.name == "Google Gemini":
                    res = self._call_gemini(provider, messages, temperature)
                else:
                    res = self._call_openrouter(provider, messages, temperature)

                self.active_index = idx  # Remember successful provider
                return res

            except Exception as e:
                last_exception = e
                if self._should_failover(e) and self.settings.provider_failover:
                    logger.warning(
                        f"[FAILOVER] Provider {provider.name} failed with error: {e}. "
                        f"Attempting failover to next provider..."
                    )
                    continue
                else:
                    # Non-transient error (or failover disabled) — raise immediately
                    raise e

        raise RuntimeError(f"All configured LLM providers failed. Last error: {last_exception}")

    def _call_openrouter(self, config: ProviderConfig, messages: List[Dict[str, str]], temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/DevasisPanda/assignRag",
            "X-Title": "AssignRAG Desktop Application"
        }

        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": temperature
        }

        response = requests.post(
            f"{config.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")

    def _call_gemini(self, config: ProviderConfig, messages: List[Dict[str, str]], temperature: float) -> str:
        """Call Google Gemini API natively via REST endpoint."""
        # Convert messages to Gemini prompt format
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        url = f"{config.base_url}/models/{config.model}:generateContent?key={config.api_key}"
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return "The information is not available in the supplied documents."
        else:
            raise Exception(f"Gemini API error {response.status_code}: {response.text}")

    def _should_failover(self, exception: Exception) -> bool:
        """Determine if an error is transient and eligible for failover (429, timeout, 5xx)."""
        err_msg = str(exception).lower()
        if "429" in err_msg or "rate limit" in err_msg or "quota" in err_msg:
            return True
        if "timeout" in err_msg or "connection" in err_msg or "connect" in err_msg:
            return True
        if any(code in err_msg for code in ["500", "502", "503", "504"]):
            return True
        return False
