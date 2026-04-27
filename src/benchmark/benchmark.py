import torch
import sys
import yaml
import os
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from dataset.processor import DatasetProcessor
from prompts.train_prompts import user_prompt, instruction_prompt
from utils.syntax import check_syntax

def load_config(path="src/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_benchmark(model, tokenizer, dataset, num_examples=200):
    results = []
    correct_syntax = 0
    
    # Switch to inference mode
    FastLanguageModel.for_inference(model)

    print(f"Running benchmark on {num_examples} examples...")
    for i in tqdm(range(min(num_examples, len(dataset)))):
        example = dataset[i]
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        
        prompt_text = user_prompt(instruction, input_text)
        messages = [
            {"role": "system", "content": instruction_prompt()},
            {"role": "user", "content": prompt_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        is_valid = check_syntax(generated_code)
        if is_valid:
            correct_syntax += 1
            
        results.append({
            "instruction": instruction,
            "input": input_text,
            "generated_code": generated_code,
            "is_valid": is_valid
        })
        
    score = (correct_syntax / len(results)) * 100 if results else 0
    return score, results

def main():
    config = load_config()
    base_model_name = config['model']['name']

    fine_tuned_model_path = config['output']['output_dir']
    
    # 1. Dataset Preparation
    # We use a dummy tokenizer first just to load and process the dataset
    print("Loading and processing dataset...")
    # Loading tokenizer from base model for dataset processing
    _, dummy_tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=config['model']['max_seq_length'],
        dtype=None,
        load_in_4bit=config['model']['load_in_4bit'],
    )
    
    dataset_processor = DatasetProcessor(config, dummy_tokenizer)
    raw_dataset = dataset_processor.load_dataset()
    # We use the same formatting/filtering logic as in training, but we'll manually handle prompts during inference
    dataset = dataset_processor.format_dataset(raw_dataset)
    
    # 2. Benchmark Base Model
    print(f"\n--- Benchmarking Base Model: {base_model_name} ---")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=config['model']['max_seq_length'],
        dtype=None,
        load_in_4bit=config['model']['load_in_4bit'],
    )
    base_score, _ = run_benchmark(base_model, dummy_tokenizer, dataset)
    print(f"Base Model Score: {base_score:.2f}%")
    
    # Clean up base model to free VRAM
    del base_model
    torch.cuda.empty_cache()

    # 3. Benchmark Fine-tuned Model
    print(f"\n--- Benchmarking Fine-tuned Model: {fine_tuned_model_path} ---")
    if os.path.exists(fine_tuned_model_path):
        ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
            model_name=fine_tuned_model_path,
            max_seq_length=config['model']['max_seq_length'],
            dtype=None,
            load_in_4bit=config['model']['load_in_4bit'],
        )
        ft_score, ft_results = run_benchmark(ft_model, ft_tokenizer, dataset)
        print(f"Fine-tuned Model Score: {ft_score:.2f}%")
        
        # Save results
        pd.DataFrame(ft_results).to_json("benchmark_results.json", index=False)
        print("\nResults saved to benchmark_results.json")
    else:
        print(f"Fine-tuned model not found at {fine_tuned_model_path}. Skipping.")

if __name__ == "__main__":
    sys.path.append(os.path.abspath("src"))
    main()
