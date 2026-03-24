import os
import ast
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Original prompts from SFT-Python-Gemma3-1b
def user_prompt(instruction: str, input_text: str) -> str:
    if input_text in ['Not applicable', '', None]:
        return f"{instruction}"
    return f"{instruction}\ninput: {input_text}"

def instruction_prompt() -> str:
    return "Act as an expert Python developer. Write a script to solve the task below. You must output only the raw, executable Python code."

def format_inference_prompt(instruction: str, input_text: str, tokenizer) -> str:
    prompt_text = user_prompt(instruction, input_text)
    messages = [
        {"role": "system", "content": instruction_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Benchmark Python Gemma-3-1b Syntax Correctness")
    parser.add_argument("--model_id", type=str, default="unsloth/gemma-3-1b-it-unsloth-bnb-4bit", help="Base model ID")
    parser.add_argument("--adapter_id", type=str, default="adson-silva/python-gemma3-1b", help="LoRA adapter ID")
    parser.add_argument("--dataset_id", type=str, default="iamtarun/python_code_instructions_18k_alpaca", help="Dataset ID")
    parser.add_argument("--num_examples", type=int, default=200, help="Number of examples to benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--output_file", type=str, default="results.csv", help="File to save results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer and model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    print(f"Loading LoRA adapters: {args.adapter_id}...")
    model = PeftModel.from_pretrained(model, args.adapter_id)
    model.eval()

    print(f"Loading dataset: {args.dataset_id}...")
    dataset = load_dataset(args.dataset_id, split="train")
    
    # Take the first N examples (the ones excluded from training)
    eval_data = dataset.select(range(min(args.num_examples, len(dataset))))

    results = []
    correct_syntax = 0

    print(f"Starting benchmark on {len(eval_data)} examples...")
    for i, example in enumerate(tqdm(eval_data)):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        
        prompt = format_inference_prompt(instruction, input_text, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # Use greedy decoding for reproducibility
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        generated_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up code blocks if model included them despite instructions
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0]
            
        is_valid = check_syntax(generated_code)
        if is_valid:
            correct_syntax += 1
            
        results.append({
            "id": i,
            "instruction": instruction,
            "input": input_text,
            "generated_code": generated_code,
            "is_valid_syntax": is_valid
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_json(args.output_file, index=False)
    
    score = (correct_syntax / len(eval_data)) * 100
    print(f"\nBenchmark results saved to {args.output_file}")
    print(f"Valid Syntax: {correct_syntax}/{len(eval_data)} ({score:.2f}%)")

if __name__ == "__main__":
    main()
