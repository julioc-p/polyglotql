#!/usr/bin/env python3
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    MistralForCausalLM,
)
import re

use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model_name = "occiglot/occiglot-7b-eu5-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)
model = MistralForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("<author>/Question-Sparql", split="validation")
dataset = dataset.filter(lambda x: x["language"] == "de")
sparql_pattern_strict = re.compile(
    r"""
    (SELECT|ASK|CONSTRUCT|DESCRIBE)
    .*?
    \}
    (
        (?:
            \s*
            (?:
                (?:(?:GROUP|ORDER)\s+BY|HAVING)\s+.+?\s*(?=\s*(?:(?:GROUP|ORDER)\s+BY|HAVING|LIMIT|OFFSET|VALUES|$)) |
                LIMIT\s+\d+ |
                OFFSET\s+\d+ |
                VALUES\s*(?:\{.*?\}|\w+|\(.*?\))
            )
        )*
    )
    """,
    re.DOTALL | re.IGNORECASE | re.VERBOSE,
)


def extract_sparql(text):
    """
    Extracts the first potential SPARQL query block from text, attempting to stop
    cleanly after the last valid SPARQL clause (e.g., LIMIT, OFFSET, ORDER BY).
    """
    code_block_match = re.search(
        r"```(?:sparql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        text_to_search = code_block_match.group(1)
    else:
        text_to_search = text
    match = sparql_pattern_strict.search(text_to_search)
    if match:
        return match.group(0).strip()
    else:
        fallback_match = re.search(
            r"(SELECT|ASK|CONSTRUCT|DESCRIBE).*?\}",
            text_to_search,
            re.DOTALL | re.IGNORECASE,
        )
        if fallback_match:
            return fallback_match.group(0).strip()
    return ""


output_data = []
batch_size = 64
for batch in dataset.iter(batch_size=batch_size):
    batch_prompts = []
    for i in range(len(batch["text_query"])):
        question = batch["text_query"][i]
        knowledge_graph = batch["knowledge_graphs"][i]
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please generate precise SPARQL queries.",
            },
            {
                "role": "user",
                "content": f"Write a SPARQL query that answers this request: '{question}' from the knowledge graph {knowledge_graph}.",
            },
        ]
        prompt_str = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        batch_prompts.append(prompt_str)
    tokenized_batch = tokenizer(
        batch_prompts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        generated_outputs = model.generate(**tokenized_batch, max_new_tokens=1024)
    for i in range(len(generated_outputs)):
        input_length = tokenized_batch.attention_mask[i].sum().item()
        generated_tokens = generated_outputs[i, input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        cleaned_sparql = extract_sparql(generated_text)
        output_data.append(
            {
                "question": batch["text_query"][i],
                "knowledge_graph": batch["knowledge_graphs"][i],
                "gold_sparql": batch["sparql_query"][i],
                "generated_sparql": cleaned_sparql,
                "raw_answer": generated_text,
            }
        )
    del tokenized_batch, generated_outputs
    torch.cuda.empty_cache()
