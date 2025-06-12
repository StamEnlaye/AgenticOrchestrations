import json
import sys
import time
import re
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score

SYS_PROMPT = """\
You are an API, not a chatbot.
Task: For each user prompt, decide if it is a QUERY that asks to locate or retrieve one or more contract clauses (return true) OR an instruction/formatting/summarizing/translating request that is NOT a retrieval query (return false).

Output format (exactly one line, valid JSON, no extra text, no markdown fences):
{"value": true}
{"value": false}
"""

CHECKER_SYS = """\
You are a verification assistant. Your job is to check whether the previous assistant correctly performed the following task:

Task: For each user prompt, decide if it is a QUERY that asks to locate or retrieve one or more contract clauses (return true)  
OR an instruction/formatting/summarizing/translating request that is NOT a retrieval query (return false).

Now you will be given:
  1) the original user prompt  
  2) the assistant's output  

Validate two things:
  • Logical correctness: that “value” matches the right answer for this prompt  
  • Formatting correctness: exactly one line of valid JSON, a single key "value" with a lowercase boolean, no extra keys, no markdown fences, no commentary

If both are OK, return the output unchanged.  
If the format is wrong but the label is correct, strip away extra characters.  
If the label is wrong, correct it to the proper boolean.  

Return exactly one line and nothing else.
"""

FEWSHOT = [
    {"role": "user", "content": "Is there a cap of liability?"},
    {"role": "assistant", "content": '{"value": true}'},
    {"role": "user", "content": "Summarize the agreement in two lines."},
    {"role": "assistant", "content": '{"value": false}'},
]

def checkerPrompt(prompt: str, previous_output: str) -> str:
    """
    Builds the user message for the checker by embedding the
    original prompt and the assistant's output.
    """
    return (
        f"Original user prompt:\n\"{prompt}\"\n\n"
        f"Assistant output:\n{previous_output}\n\n"
        f"Please apply the rules from the system message."
    )


def call_llama(model: str, messages: list) -> str:
    """Wraps ollama.chat and returns the assistant's raw text output."""
    resp = ollama.chat(
        model=model, messages=messages, options={"temperature": 0}, think=False
    )["message"]["content"]
    return resp.strip()


def parse(raw: str) -> int:
    """
    Parses {"value": true/false} from a string.
    Returns 1 for true, 0 for false, -1 if no valid JSON found.
    """
    m = re.search(r"\{[^{}]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if obj.get("value") is True:
                return 1
            if obj.get("value") is False:
                return 0
        except json.JSONDecodeError:
            pass
    token = raw.lower().split()[0]
    if token in {"true", "yes"}:
        return 1
    if token in {"false", "no"}:
        return 0
    return -1


def classify_and_check(model: str, prompt: str) -> int:
    # 1) Primary classification
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        *FEWSHOT,
        {"role": "user", "content": prompt},
    ]
    primary = call_llama(model, messages)
    print(f"First output: {primary}\n")

    # 2) Format verification
    checker_msgs = [
    {"role": "system",  "content": CHECKER_SYS},
    {"role": "user",    "content": checkerPrompt(prompt, primary)},
    ]
    verified = call_llama(model, checker_msgs)
    print(f"Second output: {verified}\n")
    # 3) Parse final JSON
    return parse(verified)


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 script.py <model_name> <prompts.json>")
        sys.exit(1)

    model_name, path = argv
    with open(path) as fh:
        rows = json.load(fh)

    t0 = time.time()
    truth, model, incorrect = [], [], []

    for row in rows:

        trueVal = 1 if row["value"] else 0
        modelAns = classify_and_check(model_name, row["prompt"])

        truth.append(trueVal)
        model.append(modelAns)
        if modelAns != trueVal:
            incorrect.append(row["prompt"])

    elapsed = time.time() - t0
    if truth:
        acc = accuracy_score(truth, model)
        prec = precision_score(truth, model, zero_division=0)
        rec = recall_score(truth, model, zero_division=0)

        print(f"\nModel: {model_name}")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall   : {rec:.2f}")
        print(f"Time total        : {elapsed:.2f}s")
        print(f"Time per prompt   : {elapsed/len(truth):.2f}s")
    else:
        print("No valid predictions to score.")

    for p in incorrect:
        print(f"Incorrect: {p}")


if __name__ == "__main__":
    main(sys.argv[1:])
