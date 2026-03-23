def user_prompt(instruction:str, input:str) -> str:
    if input in ['Not applicable', '']:
        return f"""{instruction}"""

    return f"""{instruction}\ninput: {input}"""

model_prompt = lambda output: f"""{output}"""

instruction_prompt = lambda : f"""Act as an expert Python developer. Write a script to solve the task below. You must output only the raw, executable Python code."""
