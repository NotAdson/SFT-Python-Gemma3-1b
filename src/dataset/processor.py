from datasets import load_dataset, Dataset
from src.core.base import AbstractDatasetProcessor
from src.prompts.train_prompts import user_prompt, model_prompt, instruction_prompt
from src.utils.syntax import check_syntax

class DatasetProcessor(AbstractDatasetProcessor):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def load_dataset(self, path=None):
        dataset_path = path if path else self.config['dataset']['path']
        dataset = load_dataset(dataset_path, "default", split="train")

        df = dataset.to_pandas()
        df['id'] = range(len(df))
        return Dataset.from_pandas(df)

    def format_dataset(self, dataset):
        def formatting_prompts_func(example):
            instruction = example.get('instruction', '')
            input = example.get('input', '')
            output = example.get('output', '')

            prompt_text = user_prompt(instruction, input)
            model_response_str = model_prompt(output)

            messages = [
                    {"role": "system", "content": instruction_prompt()},
                    {"role": "user", "content": prompt_text},
                    {"role": "model", "content": model_response_str},
                    ]

            return {"text": self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

        cleaned_dataset = dataset.filter(lambda example: check_syntax(example['output']))

        return cleaned_dataset.map(formatting_prompts_func)

    def split_dataset(self, dataset):
        split_config = self.config['dataset']['train_test_split']
        return dataset.train_test_split(
                test_size=split_config['test_size'], 
                shuffle=split_config['shuffle'], 
                seed=split_config['seed']
                )

    def save_dataset(self, dataset, path):
        dataset.save_to_disk(path)
