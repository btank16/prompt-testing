"""Unified LLM call interface.

Provides call_llm() which dispatches to the right provider, and
make_batch_call_fn() which wraps it into an execute_batch-compatible callable.
"""

import time

from utils import build_prompt, build_messages, normalize_response, build_response_format


def call_llm(provider, client, model, prompt, system_prompt=None,
             raw_schema=None, params=None):
    """Call any supported LLM provider and return a normalized response dict.

    Args:
        provider: "gemini", "openai", or "perplexity"
        client: The provider's client instance
        model: Model name string
        prompt: The user prompt text
        system_prompt: Optional system prompt (None to skip)
        raw_schema: Optional raw JSON schema dict
        params: Dict of provider-specific parameters passed through to the client

    Returns:
        Normalized response dict with keys: content, usage, model, citations, search_results
    """
    params = params or {}

    if provider == "gemini":
        return _call_gemini(client, model, prompt, system_prompt, raw_schema, params)
    else:
        # Perplexity and OpenAI share the same OpenAI-compatible chat/completions API
        return _call_chat_completion(provider, client, model, prompt, system_prompt, raw_schema, params)


def _call_gemini(client, model, prompt, system_prompt, raw_schema, params):
    """Gemini uses a different SDK — needs its own call path."""
    return client.generate_content(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        json_schema=raw_schema,
        **params,
    )


def _call_chat_completion(provider, client, model, prompt, system_prompt, raw_schema, params):
    """Shared path for any OpenAI-compatible chat/completions provider."""
    messages = build_messages(system_prompt, prompt)
    kwargs = {"model": model, "messages": messages}

    if raw_schema:
        kwargs["response_format"] = build_response_format(provider, raw_schema)

    # Pass through all provider-specific params (temperature, max_tokens, search filters, etc.)
    kwargs.update(params)

    raw = client.chat_completion(**kwargs)
    return normalize_response(provider, raw)


def make_batch_call_fn(provider, client, model, template, system_prompt,
                       raw_schema=None, params=None):
    """Create an execute_batch-compatible call function.

    Returns:
        call_fn(idx, row_vars) -> (idx, row_vars, response, elapsed, error)
    """
    def call_fn(idx, row_vars):
        prompt = build_prompt(template, row_vars)
        start = time.time()
        try:
            resp = call_llm(provider, client, model, prompt, system_prompt,
                            raw_schema, params)
            return idx, row_vars, resp, time.time() - start, None
        except Exception as e:
            return idx, row_vars, None, time.time() - start, str(e)

    return call_fn
