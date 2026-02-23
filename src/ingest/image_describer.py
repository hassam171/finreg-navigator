"""
Image Describer - FinReg Navigator
Generates text descriptions for charts and tables using OpenAI Vision
Saves as image chunks ready for embedding
"""

import json
import os
import time
import logging
import base64
from pathlib import Path
from datetime import datetime
import yaml
from PIL import Image
from openai import OpenAI
from openai import RateLimitError
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OPENAI_VISION_API_KEY_PAID

logger = logging.getLogger(__name__)


class ImageDescriber:
    """
    Generates descriptions for charts and tables using OpenAI Vision.
    Reads from extracted JSON, saves image chunks to chunked/ folder.
    """

    def __init__(self,
                 prompts_path="prompts/prompts.yaml",
                 output_dir="chunked/image_chunk",
                 charts_dir=None,
                 tables_dir=None,
                 model="gpt-4o-mini",
                 delay_seconds=2.0):
        self.project_root = Path(__file__).resolve().parents[2]
        self.model = model
        self.delay_seconds = delay_seconds

        # Support absolute paths for temp session dirs
        _out = Path(output_dir)
        self.output_dir = _out if _out.is_absolute() else self.project_root / _out
        os.makedirs(self.output_dir, exist_ok=True)

        # charts_dir / tables_dir overrideable so uploaded sessions point to
        # temp/{upload_id}/extracted/charts instead of the global extracted/ dir
        self.charts_dir = Path(charts_dir) if charts_dir else self.project_root / "extracted" / "charts"
        self.tables_dir = Path(tables_dir) if tables_dir else self.project_root / "extracted" / "tables"

        # Load prompts
        _prompts = Path(prompts_path)
        _prompts = _prompts if _prompts.is_absolute() else self.project_root / _prompts
        with open(_prompts, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)

        self.chart_prompt = prompts['image_description']['chart_prompt']
        self.table_prompt = prompts['image_description']['table_prompt']
        # Fallback prompt for standalone images uploaded directly (not from a PDF)
        self.standalone_prompt = prompts['image_description'].get(
            'standalone_prompt',
            "Describe this image in detail. If it contains charts, graphs, or tables, "
            "extract all key values, labels, and trends. Be concise but thorough."
        )

        # OpenAI Client
        self.client = OpenAI(
            api_key=OPENAI_VISION_API_KEY_PAID
        )

        logger.info("ImageDescriber initialized (OpenAI Vision)")
        logger.info(f"Model: {self.model}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Delay between requests: {self.delay_seconds}s")

    # ---------------------------------------------------------
    # Resize image (cost + token control)
    # ---------------------------------------------------------
    def _resize_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((1024, 1024))
        temp_path = image_path + "_resized.jpg"
        img.save(temp_path, format="JPEG", quality=85)
        return temp_path

    # ---------------------------------------------------------
    # Vision API call with retry + backoff
    # ---------------------------------------------------------
    def _call_vision(self, image_path: str, prompt: str) -> str | None:

        resized_path = self._resize_image(image_path)

        with open(resized_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        message_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ]
            }
        ]
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_payload,
                    max_tokens=200,
                    temperature=0.1
                )

                os.remove(resized_path)
                return response.choices[0].message.content

            except RateLimitError:
                wait = 6 * (attempt + 1)
                logger.warning(f"Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)

            except Exception as e:
                logger.error(f"Vision API error: {e}")
                break

        os.remove(resized_path)
        return None

    # ---------------------------------------------------------
    # Describe a standalone uploaded image (no PDF context)
    # ---------------------------------------------------------
    def describe_single_image(self, image_path: str, source_label: str = None) -> dict | None:
        """
        Describe a single image file uploaded directly (not extracted from a PDF).
        Saves a minimal image_chunk JSON to self.output_dir.

        Args:
            image_path:   Absolute path to the image file
            source_label: Optional label (defaults to filename stem)

        Returns:
            chunk dict if successful, None on failure
        """
        image_path = Path(image_path)
        label = source_label or image_path.stem

        logger.info(f"[describe_single_image] {image_path.name}")

        description = self._call_vision(str(image_path), self.standalone_prompt)

        if description is None:
            logger.warning(f"[describe_single_image] Vision API returned nothing for {image_path.name}")
            return None

        file_size_kb = round(image_path.stat().st_size / 1024, 1)

        chunk = {
            "chunk_id": f"{label}_standalone_0",
            "type": "standalone_image",
            "text": description,
            "image_path": str(image_path).replace("\\", "/"),
            "source_pdf": None,
            "page_number": None,
            "file_size_kb": file_size_kb,
            "described_at": datetime.now().isoformat()
        }

        output = {
            "pdf_name": label,
            "total_pages": 1,
            "described_at": datetime.now().isoformat(),
            "total_chunks": 1,
            "charts_described": 0,
            "tables_described": 0,
            "skipped": 0,
            "chunks": [chunk]
        }

        output_path = self.output_dir / f"{label}_image_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"[describe_single_image] Saved → {output_path}")
        return chunk

    # ---------------------------------------------------------
    # Describe single document
    # ---------------------------------------------------------
    def describe_document(self, extracted_json_path: str,
                          describe_tables: bool = True,
                          max_images: int = None) -> dict:

        logger.info("=" * 60)
        logger.info("IMAGE DESCRIBER STARTED")
        logger.info("=" * 60)

        extracted_json_path = Path(extracted_json_path)
        if not extracted_json_path.is_absolute():
            extracted_json_path = self.project_root / extracted_json_path

        with open(extracted_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pdf_name = data.get('pdf_name', 'unknown')
        total_pages = data.get('total_pages', 0)

        image_chunks = []
        chunk_counter = 0
        charts_described = 0
        tables_described = 0
        skipped = 0

        # -------------------- Charts --------------------
        all_charts = sorted(self.charts_dir.glob(f"{pdf_name}-*.png"))
        logger.info(f"Charts found: {len(all_charts)}")

        for idx, disk_file in enumerate(all_charts, start=1):

            if max_images and charts_described >= max_images:
                break

            remainder = disk_file.stem.replace(f"{pdf_name}-", "")
            page_index = int(remainder.split("-")[0])
            page_num = page_index + 1

            logger.info(f"[Chart {idx}/{len(all_charts)}] {disk_file.name}")

            description = self._call_vision(str(disk_file), self.chart_prompt)

            if description is None:
                skipped += 1
                continue

            file_size_kb = round(disk_file.stat().st_size / 1024, 1)

            image_chunks.append({
                "chunk_id": f"{pdf_name}_chart_p{page_num}_{chunk_counter}",
                "type": "chart",
                "text": description,
                "image_path": str(disk_file).replace('\\', '/'),
                "source_pdf": pdf_name,
                "page_number": page_num,
                "file_size_kb": file_size_kb,
                "described_at": datetime.now().isoformat()
            })

            chunk_counter += 1
            charts_described += 1

            time.sleep(self.delay_seconds)

        # -------------------- Tables --------------------
        if describe_tables:

            all_tables = sorted(self.tables_dir.glob(f"{pdf_name}*.png"))
            logger.info(f"Tables found: {len(all_tables)}")

            for idx, disk_file in enumerate(all_tables, start=1):

                stem = disk_file.stem
                page_part = stem.split('_page')[-1]
                page_num = int(page_part.split('_')[0])

                logger.info(f"[Table {idx}/{len(all_tables)}] {disk_file.name}")

                description = self._call_vision(str(disk_file), self.table_prompt)

                if description is None:
                    skipped += 1
                    continue

                file_size_kb = round(disk_file.stat().st_size / 1024, 1)

                image_chunks.append({
                    "chunk_id": f"{pdf_name}_table_p{page_num}_{chunk_counter}",
                    "type": "table",
                    "text": description,
                    "image_path": str(disk_file).replace('\\', '/'),
                    "source_pdf": pdf_name,
                    "page_number": page_num,
                    "file_size_kb": file_size_kb,
                    "described_at": datetime.now().isoformat()
                })

                chunk_counter += 1
                tables_described += 1

                time.sleep(self.delay_seconds)

        output = {
            "pdf_name": pdf_name,
            "total_pages": total_pages,
            "described_at": datetime.now().isoformat(),
            "total_chunks": len(image_chunks),
            "charts_described": charts_described,
            "tables_described": tables_described,
            "skipped": skipped,
            "chunks": image_chunks
        }

        output_path = self.output_dir / f"{pdf_name}_image_chunks.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved to: {output_path}")

        return output

    # ---------------------------------------------------------
    # Describe multiple documents
    # ---------------------------------------------------------
    def describe_multiple(self, extracted_json_files: list,
                          describe_tables: bool = True) -> dict:

        results = {}

        for json_file in extracted_json_files:

            json_path = Path(json_file)

            logger.info("=" * 60)
            logger.info(f"Processing: {json_path.name}")
            logger.info("=" * 60)

            try:
                result = self.describe_document(
                    str(json_path),
                    describe_tables=describe_tables
                )

                results[str(json_path)] = {
                    'success': True,
                    'total_chunks': result['total_chunks'],
                    'charts_described': result['charts_described'],
                    'tables_described': result['tables_described'],
                    'skipped': result['skipped']
                }

            except Exception as e:
                logger.exception("Error during image description")
                results[str(json_path)] = {
                    'success': False,
                    'error': str(e)
                }

        return results


# ==========================================================
# MAIN (Standalone Run)
# ==========================================================
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    logger.info("=" * 80)
    logger.info("STANDALONE IMAGE DESCRIBER RUN")
    logger.info("=" * 80)

    project_root = Path(__file__).resolve().parent.parent.parent

    extracted_dir = project_root / "extracted"
    image_chunk_dir = project_root / "chunked" / "image_chunk"
    prompts_path = project_root / "prompts" / "prompts.yaml"

    extracted_json_files = list(extracted_dir.glob("*_data.json"))

    if not extracted_json_files:
        logger.error(f"No extracted JSON files found in {extracted_dir}")
        exit()

    logger.info(f"Found {len(extracted_json_files)} extracted JSON files")

    describer = ImageDescriber(
        prompts_path=str(prompts_path),
        output_dir=str(image_chunk_dir),
        model="gpt-4o-mini",
        delay_seconds=2.0
    )

    results = describer.describe_multiple(
        extracted_json_files,
        describe_tables=True
    )

    logger.info("=" * 80)
    logger.info("STANDALONE IMAGE DESCRIBER COMPLETE")
    logger.info("=" * 80)

    successful = [r for r in results.values() if r.get("success")]
    failed = [r for r in results.values() if not r.get("success")]

    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")