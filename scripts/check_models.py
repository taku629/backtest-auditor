"""Verify the configured model IDs against the live Token Factory catalogue (optionally ping each).

    NEBIUS_API_KEY=... python scripts/check_models.py [--ping]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI  # noqa: E402

from agents.config import load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    key = cfg.api_key()
    if not key:
        print(f"{cfg.api_key_env} is not set")
        return 2
    client = OpenAI(base_url=cfg.base_url, api_key=key)
    ids = sorted(m.id for m in client.models.list().data)
    nvidia = [i for i in ids if "nemotron" in i.lower() or i.lower().startswith("nvidia/")]
    print(f"{len(ids)} models in catalogue; NVIDIA/Nemotron:\n  " + "\n  ".join(nvidia))
    ok = True
    for role, route in cfg.routes.items():
        found = route.model in ids
        ok &= found
        print(f"[{'OK' if found else 'MISSING'}] {role:10s} {route.model}")
        if found and "--ping" in sys.argv:
            r = client.chat.completions.create(model=route.model, max_tokens=16,
                                               messages=[{"role": "user", "content": "Reply with OK."}])
            print(f"           -> {r.choices[0].message.content!r} usage={r.usage}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
