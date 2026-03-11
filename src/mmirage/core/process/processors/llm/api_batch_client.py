

from typing import List, Optional, Dict, Any, Type
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
    def build_request(
        self,
        *,
        prompt: str,
        image_b64: str = None,
        media_type: str = None,
        request_id: int,
        system_prompt: str = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> dict:
        """
        Build a single API request object based on the provider.

        Args:
            text: The input text to send to the LLM.
            image_b64: Optional base64-encoded image string for multimodal models.
            request_id: Unique identifier for this request.

        Returns:
            A dict representing the API request payload.
        """
        pass


    @abstractmethod
    def submit_batches(self, output_dir: Path) -> None:
        """
        Submit batches of requests to the LLM API and save responses.

        Args:
            batches_dir: Directory containing batch request files.
            output_dir: Directory to save API responses.
        """
        pass
        
        

    @abstractmethod
    def process_dataset(self,
        *,
        nb_samples: Optional[int] = None,
    ) -> None:
        """
        Build batch JSONL files

        Writes one or more files: part_1.jsonl, part_2.jsonl, ...
        Splits by MAX_PART_SIZE_BYTES.
        """

    

    @abstractmethod
    def await_and_collect_batch_outputs(self, batches_dir: Path, output_dir: Path) -> None:
        """
        Wait for API responses and collect outputs into VariableEnvironments.

        Args:
            batches_dir: Directory containing batch request files.
            output_dir: Directory where API responses are saved.
        """        
        pass




