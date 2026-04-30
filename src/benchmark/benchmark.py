import torch
import sys
import yaml
import os
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from src.dataset.processor import DatasetProcessor
from src.prompts.train_prompts import user_prompt, instruction_prompt
from src.utils.syntax import check_syntax

def load_config(path="src/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_benchmark(model, tokenizer, dataset):
    results = []
    correct_syntax = 0
    
    FastLanguageModel.for_inference(model)

    print(f"Running benchmark on {len(dataset)} examples...")
    for i in tqdm(range(max(10, len(dataset)))):
        print(i)

def main():
    config = load_config()
    model_params = {
        "max_seq_length": config['model']['max_seq_length'],
        "dtype": None,
        "load_in_4bit": config['model']['load_in_4bit'],
    }
    
    print("Loading and processing dataset...")

    _, tokenizer = FastLanguageModel.from_pretrained(model_name=config['model']['name'], **model_params)
    dataset_processor = DatasetProcessor(config, tokenizer)
    dataset = dataset_processor.load_dataset(config['dataset']['path'])
    dataset = dataset_processor.format_dataset(dataset)
    dataset = dataset.filter(lambda example: example.get('id') < 500)
    
    models_to_test = [
        ("Base", config['model']['name']),
        ("Fine-tuned", config['output']['output_dir'])
    ]
    
    for name, path in models_to_test:
        if not os.path.exists(path) and name == "Fine-tuned":
            print(f"{name} model not found at {path}. Skipping.")
            continue
            
        print(f"\n--- Benchmarking {name} Model: {path} ---")
        model, model_tokenizer = FastLanguageModel.from_pretrained(model_name=path, **model_params)
        score, results = run_benchmark(model, model_tokenizer if name == "Fine-tuned" else tokenizer, dataset)
        print(f"{name} Model Score: {score:.2f}%")
        
        if name == "Fine-tuned":
            pd.DataFrame(results).to_json("benchmark_results.json", index=False)
            print("Results saved to benchmark_results.json")
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    sys.path.append(os.path.abspath("src"))
    main()
