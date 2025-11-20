from typing import List

import torch
from transformers import pipeline

from src.config import SUMMARIZER_MODEL_NAME


torch.set_num_threads(1) 


class BaselineSummarizer:
    """
    Simple summarizer using a Hugging Face summarization pipeline.

    This class takes a list of text chunks and returns a single summary string.
    """

    def __init__(self, model_name: str = SUMMARIZER_MODEL_NAME, max_chars: int = 4000):
        self.pipe = pipeline("summarization", model=model_name)
        self.max_chars = max_chars

    def summarize_chunks(self, chunks: List[str]) -> str:
        """
        Join all chunks and summarize them in a single call.

        For very long inputs, we truncate to `self.max_chars` characters
        to stay within model limits.
        """
        if not chunks:
            return "No content to summarize."

        full_text = "\n\n".join(chunks)

        if len(full_text) > self.max_chars:
            full_text = full_text[: self.max_chars]

        result = self.pipe(
            full_text,
            max_length=200,
            min_length=50,
            do_sample=False,
        )
        summary_text = result[0]["summary_text"]
        return summary_text
