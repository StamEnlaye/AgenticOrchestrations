import json
import sys
import time
import ollama
from sklearn.metrics import precision_score, recall_score, accuracy_score

def chat(model, prompt):
    prompt_block = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompt)])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a binary classifier. I will give you a list of 10 prompts. "
                "For each one, respond with either 'yes' if it is a query relevant to contract clause retrieval, "
                "or 'no' if it is a request, instruction, or unrelated to clause retrieval. "
                "Reply in the following format:\n"
                "1. yes\n2. no\n3. yes\n4. no\n5. yes\n6. yes\n7. yes\n8. no\n9. no\n10. yes"
            )
        },
        {
            "role": "user",
            "content": prompt_block
        }
    ]
    response = ollama.chat(model=model, messages=messages,  options={"temperature": 0})
    content = response['message']['content'].strip().lower()

    results = []
    for line in content.splitlines():
        line = line.strip()
        for i in range(1, 11):
            if line.startswith(f"{i}."):
                if "yes" in line:
                    results.append(1)
                elif "no" in line:
                    results.append(0)
                else:
                    results.append(-1)
    return results

def run_once(model_name, filtered):
    truth = []
    modelAnswers = []

    for i in range(0, len(filtered), 10):
        chunk = filtered[i:i+10]
        prompts = [item[0] for item in chunk]
        labels = [item[1] for item in chunk]

        if len(prompts) < 10:
            continue

        preds = chat(model_name, prompts)
        if len(preds) != 10:
            print("Skipping")
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
        value = str(item['value']).strip().lower()
        if value == "yes":
            filtered.append((item['prompt'], 1))
        elif value == "no":
            filtered.append((item['prompt'], 0))

    accs, precs, recs = [], [], []
    total_time = 0.0

    for run in range(10):
        print(f"--- Run {run + 1} ---")
        start_time = time.time()
        acc, prec, rec = run_once(model_name, filtered)
        duration = time.time() - start_time
        total_time += duration

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        print(f"Time:      {duration:.2f} seconds")
        print(f"Accuracy:  {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall:    {rec:.2f}")

    print("\n=== Averages Over 10 Runs ===")
    print(f"Avg Accuracy:  {sum(accs)/10:.2f}")
    print(f"Avg Precision: {sum(precs)/10:.2f}")
    print(f"Avg Recall:    {sum(recs)/10:.2f}")
    print(f"Total Time:    {total_time:.2f} seconds")
    print(f"Avg Time/Run:  {total_time/10:.2f} seconds")
