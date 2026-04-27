import yaml
import sys
import os
from transformers import AutoTokenizer
from dataset.processor import DatasetProcessor

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Ensure src is in python path if we run from project root
    sys.path.append(os.path.abspath("src"))
    
    config_path = "src/config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    config = load_config(config_path)
    
    print(f"Loading tokenizer for {config['model']['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Ensure tokenizer has chat template
    if tokenizer.chat_template is None:
        print("Warning: Tokenizer does not have a chat template. Using default if available.")

    print("Initializing DatasetProcessor...")
    processor = DatasetProcessor(config, tokenizer)
    
    print("Loading dataset...")
    raw_dataset = processor.load_dataset()
    
    print("Formatting and filtering dataset...")
    formatted_dataset = processor.format_dataset(raw_dataset)
    
    print("\n" + "="*50)
    print("PRINTING 5 EXAMPLES")
    print("="*50 + "\n")
    
    for i in range(min(5, len(formatted_dataset))):
        print(f"--- EXAMPLE {i+1} ---")
        print(formatted_dataset[i]['text'])
        print("\n" + "-"*30 + "\n")

if __name__ == "__main__":
    main()
