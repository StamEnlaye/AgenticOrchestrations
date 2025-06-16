import json
import sys
import ollama

SYS_PROMPT = (
    "You are a prompt-decomposition assistant for a construction Q&A system.\n\n"
    " Break each user prompt into FOUR lists inside one JSON object:\n"
    "   1. context     : background statements, no direct answer needed\n"
    "   2. queries      : standalone questions rewritten clearly\n"
    "   3. directives   : formatting / style requests\n"
    "   4. noise        : greetings, filler, or irrelevant words\n\n"
    "Return **ONLY** a JSON object with this schema:\n"
    "{\n"
    '  "context":   [ "..." ],\n'
    '  "queries":   [ "..." ],\n'
    '  "directives":[ "..." ],\n'
    '  "noise":     [ "..." ]\n'
    "}\n"
    "No extra keys, no prose, no markdown."
)

ONESHOT = [
    {
        "role": "user",
        "content": (
            "I'm considering a new build. I'm debating whether I should use steel or masonry frames. "
            "What are the cost differences, design duration, and which should I choose? "
            "And please summarize in a table."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            '  "context": [\n'
            '    "The user is planning a new building project.",\n'
            '    "They are evaluating steel versus masonry framing."\n'
            "  ],\n"
            '  "queries": [\n'
            '    "What is the cost difference between steel and masonry structural frames for a new building?",\n'
            '    "What is the typical design duration for steel versus masonry frames in a new building?",\n'
            '    "Which structural frame type—steel or masonry—is preferable for a new build based on cost and schedule?"\n'
            "  ],\n"
            '  "directives": [\n'
            '    "Summarize the answers in a table."\n'
            "  ],\n"
            '  "noise": []\n'
            "}"
        ),
    },
]


def call_llama(model: str, messages: list) -> str:
    return ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0},
        think=False,
    )["message"]["content"].strip()


def classify_prompt(model: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        *ONESHOT,
        {"role": "user", "content": prompt},
    ]
    return call_llama(model, messages)


def print_markdown_table(results):
    print("\n### Decomposition Results\n")
    print("| Original Prompt | Type | Content |")
    print("|-----------------|------|---------|")

    for item in results:
        prompt_text = item["prompt"].strip().replace("|", "\\|")

        # Attempt to parse model JSON
        try:
            obj = json.loads(item["response"])
        except json.JSONDecodeError:
            print(f"| {prompt_text} | error | Invalid JSON |")
            continue

        def add_row(ptype: str, text: str):
            cell = text.replace("|", "\\|").strip()
            print(f"| {prompt_text} | {ptype} | {cell} |")

        for q in obj.get("queries", []):
            add_row("query", q)

        for c in obj.get("context", []):
            add_row("context", c)

        for d in obj.get("directives", []):
            add_row("directive", d)

        for n in obj.get("noise", []):
            add_row("noise", n)


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 script.py <model_name> <prompts.json>")
        sys.exit(1)

    model_name, prompts_path = argv
    with open(prompts_path, "r") as f:
        rows = json.load(f)

    results = []
    for row in rows:
        prompt = row["prompt"]
        response = classify_prompt(model_name, prompt)
        results.append({"prompt": prompt, "response": response})

    print_markdown_table(results)


if __name__ == "__main__":
    main(sys.argv[1:])
