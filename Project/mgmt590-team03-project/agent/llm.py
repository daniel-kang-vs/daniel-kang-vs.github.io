"""OpenAI client helpers — no LangChain."""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def get_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key, timeout=45.0, max_retries=2)


def _user_message_content(
    user: str,
    images: list[dict[str, str]] | None = None,
) -> str | list[dict]:
    """Build user content — plain text or multimodal parts for vision."""
    if not images:
        return user
    parts: list[dict] = [{"type": "text", "text": user}]
    for img in images:
        mime = img.get("mime") or "image/png"
        b64 = img["b64"]
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return parts


def structured_completion(
    model_cls: Type[T],
    system: str,
    user: str,
    *,
    model: str | None = None,
    images: list[dict[str, str]] | None = None,
) -> T:
    client = get_openai_client()
    schema = model_cls.model_json_schema()
    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": (
                    f"{system}\n\nRespond with JSON only matching this schema:\n"
                    f"{json.dumps(schema, indent=2)}"
                ),
            },
            {"role": "user", "content": _user_message_content(user, images)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return model_cls.model_validate(json.loads(content))


def text_completion(
    system: str,
    user: str,
    *,
    model: str | None = None,
    images: list[dict[str, str]] | None = None,
) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _user_message_content(user, images)},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
