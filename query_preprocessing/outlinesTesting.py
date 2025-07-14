import ollama
from outlines import from_ollama, Generator
from outlines.types import JsonSchema
from pydantic import BaseModel
from typing import List

decompSys = """You are the decomposition module for a construction-contract Q&A pipeline.

Your single task: read ONE user prompt and sort every literal fragment of text into
exactly four arrays inside a single JSON object.

Arrays (include all four even when empty):

1. "context": statements describing background, scenario, or assumptions that
                  appear verbatim in the user text.
2. "queries": each explicit QUESTION rewritten for clarity while preserving
                  original technical nouns (“anchor terms”). Do not split a question
                  unless the user clearly asks more than one.
3. "directives" :instructions about format, language, or style
                  (e.g. “list the clauses”, “translate to Spanish”, “give a table”).
4. "noise" : greetings, apologies, filler words that carry no meaning.

Output rules (strict):

• Rewrite vague or subjective questions into clear, specific language using the user's original terms.
• Never invent, paraphrase, or add facts.  
• Keep anchor terms exactly (change order or casing only if needed for grammar).  
• Do not carry examples from earlier turns.  
• Return **valid JSON on one line** - no markdown, comments, or trailing commas.  
• Use straight double-quotes only.  
• Keys must appear in the order shown above.

Schema (always):

{"context":[],"queries":[],"directives":[],"noise":[]}
"""
fewshotDecomp = [
    # ── EXAMPLE 1 : combined context + multi-question ────────────────────────────
    {
        "role": "user",
        "content": (
            "I'm considering a new build. I'm debating whether to use steel or masonry frames. "
            "What are the cost differences, design duration, and which should I choose? "
            "And please summarize the results in a table."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            '  "context":    ["I am considering a new building and choosing between steel and masonry frames."],\n'
            '  "queries":    [\n'
            '    "What is the cost difference between steel and masonry structural frames?",\n'
            '    "What is the typical design duration for steel versus masonry frames?",\n'
            '    "Which frame type—steel or masonry—should be chosen for this new build?"\n'
            "  ],\n"
            '  "directives": ["Summarize the results in a table."],\n'
            '  "noise":      []\n'
            "}"
        ),
    },
    # ── EXAMPLE 2 : simple calculation (no context) ──────────────────────────────
    {
        "role": "user",
        "content": "What's 30% of 500 000?",
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            '  "context":    [],\n'
            '  "queries":    ["What is 30% of 500 000?"],\n'
            '  "directives": [],\n'
            '  "noise":      []\n'
            "}"
        ),
    },
    # ── EXAMPLE 3 : reformatting request only ────────────────────────────────────
    {
        "role": "user",
        "content": "Give me the output in a markdown table.",
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            '  "context":    [],\n'
            '  "queries":    [],\n'
            '  "directives": ["Give me the output in a markdown table."],\n'
            '  "noise":      []\n'
            "}"
        ),
    },
    # ── EXAMPLE 4 : follow-up clarification (no new context) ─────────────────────
    {
        "role": "user",
        "content": "Please give me all of the information about schedule management plan",
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            '  "context":    [],\n'
            '  "queries":    ["Give information about schedule management plan"],\n'
            '  "directives": [],\n'
            '  "noise":      ["Please"]\n'
            "}"
        ),
    },
    # ── EXAMPLE 5 : greeting + simple question (noise captured) ──────────────────
    {
        "role": "user",
        "content": "Hi there! What are liquidated damages?",
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            '  "context":    [],\n'
            '  "queries":    ["What are liquidated damages?"],\n'
            '  "directives": [],\n'
            '  "noise":      ["Hi there!"]\n'
            "}"
        ),
    },
]


class StructuredOutput(BaseModel):
    context: List[str]
    queries: List[str]
    directives: List[str]
    noise: List[str]


def generate_decomposition(model_name: str, user_prompt: str):
    client = ollama.Client()
    base_model = from_ollama(client, model_name)

    generator = Generator(base_model, StructuredOutput)

    # Convert fewshotDecomp (role-based list) into text block
    fewshot_block = ""
    for msg in fewshotDecomp:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            fewshot_block += f"User: {content}\n"
        elif role == "assistant":
            fewshot_block += f"{content}\n"

    # Combine system prompt, few-shot examples, and new user input
    full_prompt = f"{decompSys}\n\n{fewshot_block}\nUser: {user_prompt}"

    # Generate and validate
    output = generator(full_prompt)
    return StructuredOutput.model_validate_json(output)



def main():
    model_name = "llama3.2"
    user_prompt = "Hi! Can you tell me what clauses apply to late delivery? Also, please format your answer as a table."
    result = generate_decomposition(model_name, user_prompt)
    print(result)

if __name__ == "__main__":
    main()
