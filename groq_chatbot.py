"""Groq LLM interface for the FIFA World Cup 2026 Assistant.

This module is the *single* LLM interface for the whole application. Every agent
and tool that needs the language model goes through :class:`GroqChatbot` or the
module-level :func:`chat` helper.

It can still be run directly as a small CLI for quick manual testing::

    python groq_chatbot.py

The API key is resolved (in order) from:
    1. The ``GROQ_API_KEY`` environment variable.
    2. Streamlit secrets (``st.secrets["GROQ_API_KEY"]``), when running under Streamlit.
    3. A hardcoded fallback (kept only so the project runs out of the box).

NOTE: The hardcoded fallback key below is a development convenience. For any real
deployment, set ``GROQ_API_KEY`` (or use ``.streamlit/secrets.toml``) and rotate
the fallback key.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

from groq import Groq

logger = logging.getLogger(__name__)

# A bigger / more capable model than the original ``llama-3.1-8b-instant``.
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _resolve_api_key(explicit_key: Optional[str] = None) -> str:
    """Resolve the Groq API key from explicit arg, env, Streamlit secrets, or fallback."""
    if explicit_key:
        return explicit_key

    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key

    # Streamlit secrets, only if Streamlit is importable and configured.
    try:  # pragma: no cover - depends on runtime environment
        import streamlit as st  # type: ignore

        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"])
    except Exception:  # noqa: BLE001 - streamlit not installed / no secrets file
        pass

    raise RuntimeError(
        "No Groq API key found. Set the GROQ_API_KEY environment variable, add it "
        "to .streamlit/secrets.toml, or pass it explicitly."
    )


class GroqChatbot:
    """Thin, reusable wrapper around the Groq chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = Groq(api_key=_resolve_api_key(api_key))

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a single-turn prompt and return the assistant's text response."""
        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat_messages(messages, temperature, max_tokens)

    def chat_messages(
        self,
        messages: Iterable[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a list of chat messages and return the assistant's text response."""
        try:
            completion = self._client.chat.completions.create(
                messages=list(messages),
                model=self.model,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - surface a graceful message
            logger.exception("Groq API call failed")
            return f"[LLM error] {exc}"


# Module-level singleton + convenience helper -------------------------------------------------

_default_bot: Optional[GroqChatbot] = None


def get_chatbot(model: Optional[str] = None) -> GroqChatbot:
    """Return a process-wide :class:`GroqChatbot` singleton."""
    global _default_bot
    if _default_bot is None or (model is not None and model != _default_bot.model):
        _default_bot = GroqChatbot(model=model or DEFAULT_MODEL)
    return _default_bot


def chat(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Convenience one-shot chat using the singleton chatbot."""
    return get_chatbot().chat(prompt, system_prompt=system_prompt)


def _cli() -> None:
    """Tiny interactive REPL, preserved from the original script."""
    logging.basicConfig(level=logging.INFO)
    bot = get_chatbot()
    print(f"Chat with Groq [{bot.model}] (type 'exit' or 'quit' to stop)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        print(f"AI: {bot.chat(user_input)}\n")


if __name__ == "__main__":
    _cli()
