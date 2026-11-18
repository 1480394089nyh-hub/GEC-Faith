# -*- coding: utf-8 -*-
"""Minimal OpenAI-compatible chat client (pure stdlib urllib, zero dependencies).

DeepSeek / Qwen / GLM all expose OpenAI-compatible endpoints, so we uniformly
call {base_url}/chat/completions.
"""
import json
import urllib.request
import urllib.error


def chat(base_url, api_key, model, messages, temperature=0.0, timeout=180):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    return data["choices"][0]["message"]["content"]
