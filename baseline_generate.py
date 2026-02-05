#!/usr/bin/env python3
"""
Baseline generation without REFRAG - uses the decoder model directly
with all context passages to compare performance metrics.
"""

import json
import time
import argparse
import os
from typing import List, Dict
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def now_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaselineGenerator:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B", torch_compile: bool = False):
        self.device = now_device()
        print(f"[baseline] Loading model {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

        if torch_compile and hasattr(torch, 'compile'):
            print("[baseline] Applying torch.compile to model...")
            self.model = torch.compile(self.model)

        self.eos_id = self.tokenizer.eos_token_id
        print(f"[baseline] Model loaded on {self.device}")

    def format_prompt(self, question: str, passages: List[str]) -> str:
        """Format the prompt with question and all context passages."""
        context = "\n\n".join([f"Passage {i+1}:\n{p}" for i, p in enumerate(passages)])
        prompt = f"""Answer the following question based on the provided context passages.

Context:
{context}

Question: {question}

Answer:"""
        return prompt

    def generate(
        self,
        question: str,
        passages: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict:
        """Generate answer with timing metrics."""
        # Format prompt
        prompt = self.format_prompt(question, passages)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        prompt_tokens = input_ids.shape[1]
        print(f"[baseline] Prompt tokens: {prompt_tokens}")

        # Prefill - measure time to first token
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            ttft = time.time() - t0

        # Autoregressive generation - measure inter-token times
        generated = []
        ttit_list = []
        last_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                t1 = time.time()
                outputs = self.model(
                    input_ids=last_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                ttit_list.append(time.time() - t1)

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Sampling strategy (same as REFRAG)
                if temperature > 0.0:
                    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                    if top_p < 1.0:
                        # Top-p (nucleus) sampling
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        cutoff = cumulative_probs > top_p
                        cutoff[..., 1:] = cutoff[..., :-1].clone()
                        cutoff[..., 0] = False

                        sorted_probs[cutoff] = 0.0
                        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                        next_token = torch.multinomial(sorted_probs, num_samples=1)
                        next_token = sorted_indices.gather(-1, next_token)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                token_id = next_token.item()

                # Check for EOS
                if token_id == self.eos_id:
                    break

                generated.append(token_id)
                last_token = next_token

        # Decode generated tokens
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Calculate metrics
        throughput = (len(generated) / max(sum(ttit_list), 1e-6)) if ttit_list else 0.0

        return {
            "answer": answer.strip(),
            "TTFT_sec": ttft,
            "TTIT_avg_sec": float(np.mean(ttit_list)) if ttit_list else 0.0,
            "throughput_tok_per_sec": throughput,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(generated),
                "num_passages": len(passages),
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Baseline RAG generation without REFRAG")
    parser.add_argument("--json_file", type=str, required=True, help="JSON file with question-context pairs")
    parser.add_argument("--question_idx", type=int, default=0, help="Index of question to use from JSON file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B", help="Decoder model name")
    parser.add_argument("--max_new", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--output", type=str, default="", help="Output file path to save results")
    parser.add_argument("--process_all", action="store_true", help="Process all questions in JSON file")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile for the model")

    args = parser.parse_args()

    seed_everything()

    # Load model
    generator = BaselineGenerator(model_name=args.model, torch_compile=args.torch_compile)

    # Load JSON data
    print(f"[baseline] Loading data from {args.json_file}")
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_data = data.get('data', [])

    if args.process_all:
        print(f"[baseline] Processing all {len(all_data)} questions")

        for idx, item in enumerate(all_data):
            question = item.get('question', '')
            passages = item.get('context', [])

            print(f"\n[baseline] Processing question {idx + 1}/{len(all_data)}: {question[:80]}...")

            result = generator.generate(
                question=question,
                passages=passages,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            output_data = {
                "question": question,
                "passages": passages,
                **result
            }

            if args.output:
                output_file = f"{args.output}_{idx}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2)
                print(f"[baseline] Saved to {output_file}")
            else:
                print(json.dumps(output_data, indent=2))
                print("\n" + "="*80 + "\n")

        print(f"\n[baseline] Completed processing all {len(all_data)} questions")
    else:
        # Process single question
        item = all_data[args.question_idx]
        question = item.get('question', '')
        passages = item.get('context', [])

        print(f"[baseline] Processing question: {question[:80]}...")

        result = generator.generate(
            question=question,
            passages=passages,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        output_data = {
            "question": question,
            "passages": passages,
            **result
        }

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            print(f"[baseline] Saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
