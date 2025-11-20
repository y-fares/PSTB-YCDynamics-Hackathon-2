from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import LLM_MODEL_NAME


torch.set_num_threads(1) 


class LLMSummarizer:
    """
    Summarizer that uses an open-source LLM (TinyLlama) to answer
    a question based on retrieved document chunks.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        max_prompt_length: int = 512,
        max_new_tokens: int = 96,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to("cpu")

        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, query: str, chunks: List[str]) -> str:
        """
        Build a simple instruction prompt for the LLM.
        """
        if not chunks:
            return (
                "You are an assistant. There is no context available. "
                "Tell the user that no answer can be given.\n\n"
                "Answer:\n"
            )

        context = "\n\n".join(chunks)

        prompt = (
            "You are an AI assistant that answers questions based only on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "- Answer the question using ONLY the information from the context.\n"
            "- Be concise (3–5 sentences).\n"
            "- Do not invent facts.\n\n"
            "Answer:\n"
        )
        return prompt

    def summarize_with_llm(self, query: str, chunks: List[str]) -> str:
        """
        Use the LLM to produce an answer from the retrieved chunks.
        """
        prompt = self._build_prompt(query, chunks)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        text = raw_output.strip()
        marker = "Answer:"
        if marker in text:
            text = text.split(marker)[-1].strip()

        if not text:
            text = "The model did not generate an answer."

        return text
