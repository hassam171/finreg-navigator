import logging
import ollama

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str = "llama3:8b", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature

        logger.info(f"Ollama client initialized with model: {self.model}")

    def chat(self, messages: list):
        """
        messages format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature
                }
            )

            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "LLM_ERROR"