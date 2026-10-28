"""Model routing for Nebius Token Factory (OpenAI-compatible API).

Every ID is overridable by environment variable so the routing can be checked
against the live catalogue (``GET {base_url}models``) without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# USD per 1M tokens (input, output). Indicative list prices noted 2026-10-28;
# verify on the Token Factory pricing page before quoting them.
PRICING_PER_M = {
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00, 3.00),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
}


@dataclass
class ModelRoute:
    role: str
    model: str
    purpose: str
    temperature: float = 0.2
    max_tokens: int = 4096
    thinking: bool = True


@dataclass
class AgentConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "NEBIUS_API_KEY"
    routes: dict[str, ModelRoute] = field(default_factory=dict)
    request_timeout: float = 120.0
    max_retries: int = 2

    def route(self, role: str) -> ModelRoute:
        return self.routes[role]

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


def load_config() -> AgentConfig:
    env = os.environ.get
    routes = {
        "coder": ModelRoute("coder", env("CODER_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
                            "Writes and repairs the strategy function", 0.2, 8192),
        "auditor": ModelRoute("auditor", env("AUDITOR_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b"),
                              "Diagnoses failing deterministic tests, writes fix instructions", 0.1, 4096),
        "summarizer": ModelRoute("summarizer", env("SUMMARIZER_MODEL", "nvidia/Nemotron-3_5-Lightning"),
                                 "Writes the final audit summary", 0.2, 800, thinking=False),
    }
    return AgentConfig(base_url=env("NEBIUS_BASE_URL", DEFAULT_BASE_URL), routes=routes)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    p = PRICING_PER_M.get(model)
    if p is None:
        return None
    return prompt_tokens / 1e6 * p[0] + completion_tokens / 1e6 * p[1]


def use_mock() -> bool:
    v = os.environ.get("USE_MOCK")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "on")
    return not os.environ.get("NEBIUS_API_KEY")
