from __future__ import annotations

import json
import os
import sys

import requests


def _headers() -> dict[str, str]:
    token = (
        os.getenv("ENERGY_LOCAL_LLM_API_KEY", "").strip()
        or os.getenv("LM_API_TOKEN", "").strip()
        or os.getenv("LM_STUDIO_API_KEY", "").strip()
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def main() -> int:
    base_url = (os.getenv("ENERGY_LOCAL_LLM_BASE_URL", "").strip() or "http://127.0.0.1:1234/v1").rstrip("/")
    print(f"base_url: {base_url}")
    token_present = bool(_headers())
    print(f"auth: {'present' if token_present else 'missing'} (set ENERGY_LOCAL_LLM_API_KEY or LM_API_TOKEN)")

    try:
        resp = requests.get(f"{base_url}/models", headers=_headers(), timeout=5)
        print(f"/models status: {resp.status_code}")
        if resp.status_code != 200:
            print(resp.text[:800])
            return 2
        data = resp.json()
    except Exception as exc:
        print(f"ERROR calling /models: {exc}")
        return 2

    models = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
    print(f"models: {models[:6]}")
    if not models:
        print("No models returned. Load a model in LM Studio and retry.")
        return 3

    chosen = os.getenv("ENERGY_LOCAL_LLM_MODEL", "").strip() or str(models[0])
    print(f"chosen model: {chosen}")

    payload = {
        "model": chosen,
        "messages": [{"role": "user", "content": "Say 'ok'."}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=_headers(), json=payload, timeout=20)
        print(f"/chat/completions status: {resp.status_code}")
        print(resp.text[:800])
        return 0 if resp.status_code == 200 else 4
    except Exception as exc:
        print(f"ERROR calling /chat/completions: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())

