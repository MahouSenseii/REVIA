"""LLM provider adapters for the V2.3 ProviderRegistry.

Each adapter exposes a uniform :class:`ProviderAdapter` interface so the
:class:`ModelRouter` can swap between local and cloud backends at runtime
without any agent caring.

Public surface::

    ProviderAdapter   — abstract base
    OpenAICompatAdapter — shared OpenAI-style /v1/chat/completions client
    LlamaCppAdapter   — local llama.cpp HTTP server (port 8080)
    LmStudioAdapter   — local LM Studio (port 1234)
    KoboldCppAdapter  — local koboldcpp (port 5001)
    TabbyApiAdapter   — local TabbyAPI (port 5000)
    VllmAdapter       — local vLLM (port 8000)
    OllamaAdapter     — local Ollama via native /api/chat (port 11434)
    OpenAIAdapter     — OpenAI cloud (api.openai.com), key required
"""
from __future__ import annotations

from .base import ProviderAdapter, ProviderInfo
from .openai_compat import OpenAICompatAdapter
from .llamacpp import LlamaCppAdapter
from .lmstudio import LmStudioAdapter
from .koboldcpp import KoboldCppAdapter
from .tabbyapi import TabbyApiAdapter
from .vllm_adapter import VllmAdapter
from .ollama import OllamaAdapter
from .openai_cloud import OpenAIAdapter

__all__ = [
    "KoboldCppAdapter",
    "LlamaCppAdapter",
    "LmStudioAdapter",
    "OllamaAdapter",
    "OpenAIAdapter",
    "OpenAICompatAdapter",
    "ProviderAdapter",
    "ProviderInfo",
    "TabbyApiAdapter",
    "VllmAdapter",
]
