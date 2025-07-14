import json
import sys
import ollama

SYS_PROMPT = (
    "You are a prompt-decomposition assistant for a construction Q&A system.\n\n"

    "Your ONLY job: read ONE user prompt and place EVERY fragment of text into "
    "exactly one of four lists in a single JSON object.\n\n"

    "Lists (all MUST appear, even if empty):\n"
    "  1. context    - background or situational statements that are literally present in the prompt. "
    "                  • NEVER invent or paraphrase facts that are not written.\n"
    "  2. queries     - stand-alone, clearly rewritten questions that are *explicitly asked* in the prompt. "
    "                  • Rewrite for clarity but DO NOT add new details or split unless there are clearly multiple questions.\n"
    "  3. directives  - formatting or style instructions (e.g. “put it in a table”, “summarise in 2 lines”).\n"
    "  4. noise       - greetings, pleasantries, or filler words that have no informational value.\n\n"

    "STRICT RULES:\n"
    "• Do NOT copy examples from earlier turns; answer only for THIS prompt.\n"
    "• Do NOT invent context sentences; if the prompt lacks background, leave \"context\": [].\n"
    "• Do NOT inject default examples like steel-vs-masonry unless they appear in the prompt.\n"
    "• Preserve the user's nouns/terms (anchor terms) exactly.\n"
    "• The output must be valid JSON. No extra keys, comments, or markdown.\n\n"

    "Schema to return *every time*:\n"
    "{\n"
    "  \"context\":    [ \"...\" ],\n"
    "  \"queries\":    [ \"...\" ],\n"
    "  \"directives\": [ \"...\" ],\n"
    "  \"noise\":      [ \"...\" ]\n"
    "}\n"
)


FEWSHOT = [

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
    '  ],\n'
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
  "content": "Put that in a table.",
},
{
  "role": "assistant",
  "content": (
    "{\n"
    '  "context":    [],\n'
    '  "queries":    [],\n'
    '  "directives": ["Put that in a table."],\n'
    '  "noise":      []\n'
    "}"
  ),
},

# ── EXAMPLE 4 : follow-up clarification (no new context) ─────────────────────
{
  "role": "user",
  "content": "Can you explain that further?",
},
{
  "role": "assistant",
  "content": (
    "{\n"
    '  "context":    [],\n'
    '  "queries":    ["Can you explain that further?"],\n'
    '  "directives": [],\n'
    '  "noise":      []\n'
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
        *FEWSHOT,
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
