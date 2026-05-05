import sys
import os
import yaml
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from src.dataset.processor import DatasetProcessor
from src.utils.syntax import check_syntax


def load_config(path="src/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, tokenizer, dataset, n_samples: int) -> float:
    """Generate code for each sample and score by valid Python syntax."""
    FastLanguageModel.for_inference(model)
    samples = dataset.select(range(min(n_samples, len(dataset))))
    points = 0

    for example in tqdm(samples, desc="Evaluating", leave=False):
        encodings = tokenizer(
            example["text"], return_tensors="pt", truncation=True
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens=256,
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

    print("Loading and formatting dataset...")
    processor = DatasetProcessor(config, tokenizer)
    dataset = processor.load_dataset()
    dataset = processor.format_dataset(dataset)
    dataset = dataset.filter(lambda ex: ex.get("id", 0) < n_samples)

    scores = {}

    # --- Base model ---
    print("\n[1/2] Benchmarking base model...")
    scores["Base"] = evaluate(model, tokenizer, dataset, n_samples)

    # --- Fine-tuned model (load local LoRA adapters onto the same base) ---
    if os.path.isdir(adapter_path):
        print(f"\n[2/2] Loading LoRA adapters from '{adapter_path}'...")
        model = FastLanguageModel.get_peft_model(model, adapter_path)
        scores["Fine-tuned"] = evaluate(model, tokenizer, dataset, n_samples)
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
