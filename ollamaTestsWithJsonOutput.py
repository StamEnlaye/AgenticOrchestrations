import json
import sys
import time
import argparse
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score


SYSTEM_MSG = (
    # --- task -------------------------------------------------------------- #
    "You are an API, **not** a chatbot.\n"
    "Task: For each user prompt decide if it is a QUERY that asks to locate or "
    "retrieve one or more contract clauses (return true) OR an instruction / "
    "formatting / summarising / translating request that is *not* a retrieval "
    "query (return false).\n\n"
    # --- output rules ------------------------------------------------------ #
    'Return **exactly one line** of JSON with a single key named "value" and a '
    "lower-case boolean.  Only allowed forms:\n"
    '  {"value": true}\n'
    '  {"value": false}\n\n'
    "No other keys. No arrays. No markdown fences, numbers, tables, or text."
)


FEWSHOT = [
    {"role": "user",      "content": "Is there a cap of liability?"},
    {"role": "assistant", "content": '{"value": true}'},
]


def getAns(model: str, prompt: str) -> int:
    """Return 1 (true), 0 (false), or -1 if output is malformed."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        *FEWSHOT,
        {"role": "user", "content": prompt},
    ]

    resp = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0}
    )["message"]["content"].strip()

    if resp.startswith("```"):
        resp = resp.strip("` \n")

    try:
        val = json.loads(resp)["value"]
        return 1 if val is True else 0
    except Exception:
        print("Unrecognized output:", resp)
        return -1


def main(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on query vs non-query classification."
    )
    parser.add_argument("model", help="Ollama model name (e.g. mistral)")
    parser.add_argument("json_file", help="Path to prompts JSON")
    args = parser.parse_args(argv)

    with open(args.json_file, "r") as fh:
        rows = json.load(fh)

    truth, model, skipped = [], [], 0
    t0 = time.time()

    for row in rows:
        if not isinstance(row.get("value"), bool):
            skipped += 1
            continue

        trueVal = 1 if row["value"] else 0
        modelAns = getAns(args.model, row["prompt"])

        if modelAns == -1:
            skipped += 1
            continue

        truth.append(trueVal)
        model.append(modelAns)

    elapsed = time.time() - t0

    if truth:
        acc  = accuracy_score(truth, model)
        prec = precision_score(truth, model, zero_division=0)
        rec  = recall_score(truth, model, zero_division=0)

        print(f"\nModel: {args.model}")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall   : {rec:.2f}")
        print(f"Total time          : {elapsed:.2f} s")
        print(f"Average time/prompt : {elapsed/len(truth):.2f} s")
        if skipped:
            print(f"Skipped rows        : {skipped}")
    else:
        print("No valid predictions to score.")

if __name__ == "__main__":
    main(sys.argv[1:])
