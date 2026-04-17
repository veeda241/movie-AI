from __future__ import annotations

import json
import os
from typing import Any

from huggingface_hub import InferenceClient

DEFAULT_HF_TEXT_MODEL = os.environ.get("HF_TEXT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
MAX_NEW_TOKENS = 1000
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 20

def _build_client() -> InferenceClient:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Set a valid Hugging Face token and rerun the pipeline.")

    return InferenceClient(api_key=token, timeout=120)


def _extract_generated_text(response_data: Any) -> str:
    choices = getattr(response_data, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise RuntimeError("HF chat completion returned an unexpected response shape.")

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        chunks.append(str(item["text"]))
                    elif "content" in item:
                        chunks.append(str(item["content"]))
                    else:
                        chunks.append(json.dumps(item))
                else:
                    chunks.append(str(item))
            return "".join(chunks)

        return str(content)

    if isinstance(response_data, list):
        if not response_data:
            raise RuntimeError("HF text generation returned an empty response.")
        first_item = response_data[0]
        if isinstance(first_item, dict) and "generated_text" in first_item:
            return str(first_item["generated_text"])
        return json.dumps(first_item)

    if isinstance(response_data, dict):
        if "error" in response_data:
            raise RuntimeError(str(response_data["error"]))
        if "generated_text" in response_data:
            return str(response_data["generated_text"])
        return json.dumps(response_data)

    raise RuntimeError(f"Unexpected HF text generation response type: {type(response_data).__name__}")


def _extract_json_candidate(text: str) -> str | None:
    start_indexes = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not start_indexes:
        return None

    start_index = min(start_indexes)
    end_index = max(text.rfind("}"), text.rfind("]"))
    if end_index <= start_index:
        return None

    return text[start_index : end_index + 1]


def call_hf_json(prompt: str) -> Any:
    client = _build_client()
    full_prompt = (
        f"{prompt}\n\nReturn only valid JSON. Do not include markdown fences, code blocks, or commentary."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a strict JSON generator. Output only raw JSON."},
                    {"role": "user", "content": full_prompt},
                ],
                model=DEFAULT_HF_TEXT_MODEL,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                top_p=1.0,
            )
            generated_text = _extract_generated_text(response)
            generated_text = generated_text.strip()
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 503 and attempt < MAX_RETRIES:
                print(
                    f"[HFTextClient] model loading, retrying in {RETRY_DELAY_SECONDS} seconds "
                    f"({attempt}/{MAX_RETRIES})"
                )
                import time

                time.sleep(RETRY_DELAY_SECONDS)
                continue

            raise RuntimeError(f"HF text generation failed: {exc}") from exc

        generated_text = generated_text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(generated_text)
        except json.JSONDecodeError:
            candidate = _extract_json_candidate(generated_text)
            if candidate is not None:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"HF model returned invalid JSON: {generated_text}") from exc

            raise ValueError(f"HF model returned invalid JSON: {generated_text}")

    raise RuntimeError("HF text generation failed after retries.")
