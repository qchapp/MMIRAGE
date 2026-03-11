from pathlib import Path

import anthropic

from mmirage.core.process.processors.llm.api_batch_client import APIBatchClient



class AnthropicBatchClient(APIBatchClient):
    def __init__(self, model_name: str, api_key: str, output_dir: Path):
        super().__init__(model_name=model_name, api_key=api_key, provider="anthropic")

        if not self.api_key: 
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. Please export it before running."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.output_dir = output_dir