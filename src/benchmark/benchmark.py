import sys
import os
import yaml
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.dataset.processor import DatasetProcessor
from src.prompts.train_prompts import user_prompt, instruction_prompt
from src.utils.syntax import check_syntax


def load_config(path="src/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_inference_prompt(example, tokenizer) -> str:
    """Build a prompt with only the user turn so the model must generate the answer."""
    messages = [
        {"role": "system", "content": instruction_prompt()},
        {"role": "user", "content": user_prompt(
            example.get("instruction", ""),
            example.get("input", ""),
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def evaluate(model, tokenizer, dataset) -> float:
    """Generate code for each sample and score by valid Python syntax."""
    model.eval()
    points = 0

    for example in tqdm(dataset, desc="Evaluating"):
        prompt = build_inference_prompt(example, tokenizer)
        encodings = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens=128,
                do_sample=False,          # greedy — fastest + deterministic
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][encodings.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        if check_syntax(generated):
            points += 1


    return points / len(dataset) * 100


def main():
    config = load_config()
    adapter_path = config["output"]["output_dir"]
    n_samples = config.get("benchmark", {}).get("n_samples", 500)

    model_name = config["model"]["name"]
    load_in_4bit = config["model"].get("load_in_4bit", True)

    print("Loading base model and tokenizer using HuggingFace Transformers...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure quantization if required
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config["training"].get("bf16", False) else torch.float16,
        )
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if config["training"].get("bf16", False) else torch.float16,
    )

    print(f"Loading and filtering dataset (First {n_samples} examples, then syntax filter)...")
    processor = DatasetProcessor(config, tokenizer)
    raw_dataset = processor.load_dataset()
    
    # First get the subset of the dataset
    subset_dataset = raw_dataset.select(range(min(n_samples, len(raw_dataset))))
    
    # Then filter by valid syntax on the expected output
    filtered_dataset = subset_dataset.filter(lambda ex: check_syntax(ex["output"]))
    print(f"Dataset filtered: {len(filtered_dataset)} examples remaining from the initial {n_samples}.")

    scores = {}

    # --- Base model ---
    print(f"\n[1/2] Benchmarking base model on {len(filtered_dataset)} samples...")
    scores["Base"] = evaluate(base_model, tokenizer, filtered_dataset)

    # --- Fine-tuned: attach local LoRA adapters ---
    if os.path.isdir(adapter_path):
        print(f"\n[2/2] Loading LoRA adapters from '{adapter_path}' using PEFT...")
        fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Benchmarking fine-tuned model on {len(filtered_dataset)} samples...")
        scores["Fine-tuned"] = evaluate(fine_tuned_model, tokenizer, filtered_dataset)
    else:
        print(f"\n[2/2] Adapter path '{adapter_path}' not found — skipping fine-tuned eval.")

    # --- Final report ---
    print("\n" + "=" * 40)
    print("  BENCHMARK RESULTS")
    print("=" * 40)
    for name, score in scores.items():
        print(f"  {name:12s}: {score:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath("."))
    main()
