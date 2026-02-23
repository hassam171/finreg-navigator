import logging
import ollama
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaVisionClient:
    """
    Vision-capable Ollama client for models like:
    - llava
    - llama3.2-vision
    - qwen2.5vl
    """

    def __init__(self, model: str = "llava", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

        logger.info(f"Ollama Vision client initialized with model: {self.model}")

    def describe_image(self, image_path: str, prompt: str) -> str | None:
        """
        Send image + prompt to Ollama vision model.
        """

        path = Path(image_path)

        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(path)]
                    }
                ],
                options={
                    "temperature": self.temperature
                }
            )

            return response["message"]["content"]

        except Exception:
            logger.exception("Ollama vision call failed")
            return None