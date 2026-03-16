from openai import OpenAI
import json, time, re
from tqdm import tqdm
from typing import List, Tuple, Optional, Type
import json
from pathlib import Path
from pydantic import BaseModel

from mmirage.core.process.processors.llm.api_batch_client import APIBatchClient
from mmirage.core.process.processors.llm.api_utils import get_media_type, load_data, encode_image_to_base64
from mmirage.core.process.processors.llm.config import LLMOutputVar
from mmirage.core.process.variables import VariableEnvironment



# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

MAX_TOKENS = 1000
# 50 MB per batch part is well below OpenAI limits and avoids failures
MAX_PART_SIZE_BYTES = int(0.05 * 1024 ** 3)




class OpenAIBatchClient(APIBatchClient):
    def __init__(self, model_name: str, api_key: str, output_dir: Path):
        super().__init__(model_name=model_name, api_key=api_key, provider="openai")

        if not self.api_key: 
            raise SystemExit(
                "OPENAI_API_KEY is not set. Please export it before running."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.output_dir = output_dir
        self.batches_dir = output_dir / "batches" #TODO to implement in process_dataset and submit_batches


    # ---------------------------------------------------------------------
    # Request builder 
    # ---------------------------------------------------------------------


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
        """Build a single OpenAI Batch API request object.

        Args:
            prompt: The fully-rendered user prompt (Jinja2 already applied).
            image_b64: Optional base64-encoded image for multimodal requests.
            media_type: MIME type of the image (e.g., "image/jpeg").
            request_id: Unique identifier used as custom_id.
            system_prompt: Optional system message prepended to the conversation.
            output_schema: Optional Pydantic model used to enforce a JSON response
                           via OpenAI's structured-output ``response_format``.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_b64 is not None and media_type is not None:
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                },
            ]
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        body: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
        }

        if output_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "strict": True,
                    "schema": output_schema.model_json_schema(),
                },
            }

        return {
            "custom_id": f"request-{request_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    


    # ---------------------------------------------------------------------
    # Batch construction TODO
    # ---------------------------------------------------------------------




    def process_dataset(self,
        batch: List[Tuple[str, Tuple[Tuple[str, str], ...]]],
    ) -> None:
        """
        Build batch JSONL files for OpenAI Batch API. Writes one or more files: part_1.jsonl, part_2.jsonl, ...
        Splits by MAX_PART_SIZE_BYTES.

        Args:
            batch: List of (prompt_text, ((encoded_image1, media_type1), (encoded_image2, media_type2), ...))
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        part_idx = 1
        bytes_in_part = 0

        part_path = self.output_dir / f"part_{part_idx}.jsonl"
        part_file = part_path.open("w", encoding="utf-8")

        for i, (text, encoded_images) in tqdm(
            enumerate(batch, start=1),
            total=len(batch),
            desc="Building batch requests",
        ):

            # Enforce one image per request TODO : allow more than one image
            image_b64 = encoded_images[0][0]
            media_type = encoded_images[0][1]
            req = self.build_request(
                prompt=text, 
                image_b64=image_b64,
                media_type=media_type,
                request_id=i,
                system_prompt=None,# TODO
                output_schema=None,# TODO 
                )

            line = json.dumps(req, ensure_ascii=False) + "\n"
            size = len(line.encode("utf-8"))

            if bytes_in_part + size > MAX_PART_SIZE_BYTES:
                part_file.close()
                part_idx += 1
                bytes_in_part = 0
                part_path = self.output_dir / f"part_{part_idx}.jsonl"
                part_file = part_path.open("w", encoding="utf-8")

            part_file.write(line)
            bytes_in_part += size

        part_file.close()
        print(f"[DONE] Created {part_idx} batch file(s) in {self.output_dir}")





    # ---------------------------------------------------------------------
    # Batch submission
    # ---------------------------------------------------------------------


    def submit_batches(self, batches_dir: Path) -> None:
        """Submit batch files to OpenAI Batch API."""
        
        parts = sorted(batches_dir.glob("*.jsonl"))

        if not parts:
            raise SystemExit(f"No batch files found in {batches_dir}")


        # submit batches sequentially with progress bar; persist batch IDs for reproducibility
        for part in tqdm(parts, desc="Submitting batches"):
            batch_id_file = self.output_dir / f"batch_id_{part.name}.txt"

            # Skip if already submitted
            if batch_id_file.exists():
                print(f"[SKIP] {part.name} already submitted")
                continue

            # Upload batch input file
            with part.open("rb") as fh:
                uploaded = self.client.files.create(
                    file=fh,
                    purpose="batch",
                )

            # Create batch job
            batch = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"Dataset augmentation - {part.name}",
                },
            )

            # Persist batch ID (critical for reproducibility)
            batch_id_file.write_text(batch.id)


            print(f"[SUBMITTED] {part.name} → batch_id={batch.id}")
        
        print("All batches submitted.") 





    # ---------------------------------------------------------------------
    # Collect Outputs
    # ---------------------------------------------------------------------
    

    def __wait_for_output(self, batch_id: str, max_wait_s: int = 86400, poll_s: int = 30):
        waited = 0
        while True:
            b = self.client.batches.retrieve(batch_id)
            print(f"[{batch_id}] status={b.status} out={b.output_file_id} err={b.error_file_id}")
            if b.output_file_id:
                return b
            if b.status in ("failed", "cancelled", "expired"):
                raise SystemExit(f"Batch ended with status: {b.status}")
            time.sleep(poll_s)
            waited += poll_s
            if waited >= max_wait_s:
                raise SystemExit("Timed out waiting for output_file_id")

    def __part_number_from_filename(self, p: Path) -> int:
        m = re.search(r"batch_id_part_(\d+)\.jsonl\.txt$", p.name)
        return int(m.group(1)) if m else 0
    
    def __extract_messages(api_responses: List[dict]) -> List[str]:
        return [
            rec["response"]["body"]["choices"][0]["message"]["content"].strip()
            for rec in api_responses
    ]

    def __save_part_output(self, b, part_num: int, output_dir : Path) -> List[dict]:
        text = self.client.files.content(b.output_file_id).text
        part_path = output_dir / f"api_response_part_{part_num}.jsonl"
        part_path.write_text(text)
        print(f"[saved] {part_path}")
        return [json.loads(line) for line in text.splitlines() if line.strip()]


    def await_and_collect_batch_outputs(self) -> None:
        """Wait for batch completions and download outputs."""
        batch_id_files = sorted(self.output_dir.glob("batch_id_*.txt"))

        if not batch_id_files:
            raise SystemExit(f"No batch ID files found in {self.output_dir}")


        all_records = []
        total_prompt = total_completion = 0


        for id_file in batch_id_files:
            part_num = self.__part_number_from_filename(id_file)
            batch_id = id_file.read_text().strip()

            b = self.__wait_for_output(batch_id)
            records = self.__save_part_output(b, part_num)
            all_records.extend(records)

            # accumulate actual usage
            for rec in records:
                usage = rec.get("response", {}).get("body", {}).get("usage", {})
                total_prompt += int(usage.get("prompt_tokens", 0))
                total_completion += int(usage.get("completion_tokens", 0))


        # merged outputs
        all_path = self.output_dir / "api_response_all.jsonl"
        with all_path.open("w", encoding="utf-8") as fout:
            for rec in all_records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[merged] {all_path} ({len(all_records)} responses)")


        # optional: also save the plain texts
        texts_path = self.output_dir / "messages_all.txt"
        with texts_path.open("w", encoding="utf-8") as ftxt:
            for msg in self.__extract_messages(all_records):
                ftxt.write(msg + "\n\n")
        print(f"[texts]  {texts_path}")




        for batch_id_file in tqdm(batch_id_files, desc="Waiting for batches"):
            batch_id = batch_id_file.read_text().strip()
            batch = self.__wait_for_output(batch_id)

            # Download output file
            output_path = self.output_dir / f"output_{batch_id}.jsonl"
            with output_path.open("wb") as fh:
                self.client.files.download(batch.output_file_id, fh)

            print(f"[DOWNLOADED] Batch {batch_id} output to {output_path}")


