"""Model access. Bedrock (Converse API) by default; any object with complete() works."""

from __future__ import annotations

import os
from typing import Protocol

import boto3
from botocore.config import Config


class Llm(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str: ...


class BedrockLlm:
    def __init__(self, model_id: str | None = None, region: str | None = None):
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(read_timeout=60, retries={"max_attempts": 3, "mode": "adaptive"}),
        )

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.2},
        )
        return "".join(p.get("text", "") for p in resp["output"]["message"]["content"]).strip()
