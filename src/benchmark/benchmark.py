import sys
import os
import yaml
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
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


def evaluate(model, tokenizer, raw_dataset, n_samples: int) -> float:
    """Generate code for each sample and score by valid Python syntax."""
    FastLanguageModel.for_inference(model)

    samples = raw_dataset.select(range(min(n_samples, len(raw_dataset))))
    points = 0

    for example in tqdm(samples, desc="Evaluating", leave=False):
        prompt = build_inference_prompt(example, tokenizer)
        encodings = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).to("cuda")

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

    return points / len(samples) * 100


def main():
    config = load_config()
    adapter_path = config["output"]["output_dir"]
    n_samples = config.get("benchmark", {}).get("n_samples", 200)

    model_params = {
        "model_name": config["model"]["name"],
        "max_seq_length": config["model"]["max_seq_length"],
        "dtype": None,
        "load_in_4bit": config["model"]["load_in_4bit"],
    }

    print("Loading base model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(**model_params)

    # Load raw dataset once — inference prompt is built on-the-fly in evaluate()
    print("Loading dataset...")
    processor = DatasetProcessor(config, tokenizer)
    raw_dataset = processor.load_dataset()
    raw_dataset = raw_dataset.filter(lambda ex: ex.get("id", 0) < n_samples)

    scores = {}

    # --- Base model ---
    print(f"\n[1/2] Benchmarking base model on {len(raw_dataset)} samples...")
    scores["Base"] = evaluate(model, tokenizer, raw_dataset, n_samples)

    # --- Fine-tuned: attach local LoRA adapters to the already-loaded base ---
    if os.path.isdir(adapter_path):
        print(f"\n[2/2] Loading LoRA adapters from '{adapter_path}'...")
        model.load_adapter(adapter_path)
        print(f"Benchmarking fine-tuned model on {len(raw_dataset)} samples...")
        scores["Fine-tuned"] = evaluate(model, tokenizer, raw_dataset, n_samples)
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
