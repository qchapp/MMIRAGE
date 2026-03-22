

from typing import List, Optional, Dict, Any, Tuple, Type
from pydantic import BaseModel
from pathlib import Path


from mmirage.core.process.variables import VariableEnvironment
from abc import ABC, abstractmethod

class APIBatchClient(ABC):

    def __init__(self, model_name: str, api_key: str, provider: str):
        self.model_name = model_name
        self.api_key = api_key
        self.provider = provider


    @abstractmethod
    def submit_batches(self) -> None:
        """
        Submit batches of requests to the LLM API and save responses.
        """
        pass
      
    @abstractmethod
    def process_dataset(self,
        batch: List[Tuple[str, Tuple[Tuple[str, str], ...]]],
    ) -> None:
        """
        Build batch JSONL files for OpenAI Batch API. Writes one or more files: part_1.jsonl, part_2.jsonl, ...
        Splits by MAX_PART_SIZE_BYTES.

        Args:
            batch: List of (prompt_text, ((encoded_image1, media_type1), (encoded_image2, media_type2), ...))
        """
        pass

    @abstractmethod
    def await_and_collect_batch_outputs(self) -> None:
        """
        Wait for API responses and collect outputs into VariableEnvironments.
        """        
        pass