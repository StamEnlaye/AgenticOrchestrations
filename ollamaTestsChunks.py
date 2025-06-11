import json
import sys
import time
import ollama
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Prompt instructions
message = (
    "You are an API, not a chatbot.\n"
    "Task: I will give you a list of prompts. For each user prompt decide if it is a QUERY that asks to locate or "
    "retrieve one or more contract clauses (return yes) OR an instruction / "
    "formatting / summarising / translating request that is *not* a retrieval "
    "query (return no).\n\n"
    "Return exactly 10 lines, numbered 1 through 10, with each line containing either 'yes' or 'no'.\n"
    "Only allowed forms:\n"
    "  1. yes\n"
    "  2. no\n"
    "  3. yes\n"
    "  ... and so on up to line 10.\n"
    "No JSON. No extra commentary. No markdown, bullets, or formatting — just the numbered list of responses."
)
FEWSHOT = [
    {
        "role": "user",
        "content": (
            "1. Summarize the information in 2 sentences\n"
            "2. Is there a cap of liability?"
        )
    },
    {
        "role": "assistant",
        "content": (
            "1. no\n"
            "2. yes"
        )
    }
]


def chat(model, prompt, size):
    prompt_block = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompt)])
    messages = [
        {"role": "system", "content": message},
        *FEWSHOT,
        {"role": "user", "content": prompt_block}
    ]
    response = ollama.chat(model=model, messages=messages, options={"temperature": 0}, think=True)
    content = response['message']['content'].strip().lower()
    print(f"\nAnswer: {content}")
    results = []
    for i in range(1, size + 1):
        match = next((line for line in content.splitlines() if line.startswith(f"{i}.")), None)
        if match:
            if "yes" in match:
                results.append(1)
            elif "no" in match:
                results.append(0)
            else:
                results.append(-1)
        else:
            results.append(-1)
    return results

def run_once(model_name, filtered, size):
    truth = []
    modelAnswers = []

    for i in range(0, len(filtered), size):
        chunk = filtered[i:i+size]
        if len(chunk) < size:
            continue
        prompts = [item[0] for item in chunk]
        labels = [item[1] for item in chunk]

        preds = chat(model_name, prompts, size)
        if len(preds) != size:
            print("Skipping chunk due to unexpected output length.")
            continue

        for pred, label in zip(preds, labels):
            if pred != -1:
                truth.append(label)
                modelAnswers.append(pred)

    acc = accuracy_score(truth, modelAnswers)
    prec = precision_score(truth, modelAnswers)
    rec = recall_score(truth, modelAnswers)
    return acc, prec, rec

if __name__ == "__main__":
    model_name = sys.argv[1]
    json_path = sys.argv[2]

    with open(json_path, 'r') as f:
        data = json.load(f)

    filtered = []
    for item in data:
        value = item['value']
        if isinstance(value, bool):
            label = 1 if value else 0
            filtered.append((item["prompt"], label))
            continue

    start_time = time.time()
    acc, prec, rec = run_once(model_name, filtered, 10)
    duration = time.time() - start_time

    print(f"Runtime:   {duration:.2f} seconds")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
