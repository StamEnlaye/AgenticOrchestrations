import json
import sys
import ollama
from sklearn.metrics import precision_score, recall_score, accuracy_score

def chat(model, prompt):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a binary classifier. Reply with 'yes' if the input is a query "
                "relevant to contract clause retrieval, and 'no' if it is a request, instruction, "
                "or unrelated to clause retrieval. Respond only with 'yes' or 'no'."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = ollama.chat(model=model, messages=messages)
    content = response['message']['content'].strip().lower()

    if content.startswith("yes") or content.startswith("Yes"):
        return 1
    elif content.startswith("no") or content.startswith("No"):
        return 0
    else:
        print("skipping")
        return -1

if __name__ == "__main__":
    model_name = sys.argv[1]
    json_path = sys.argv[2]
    truth = [] # ground truth list
    modelAnswers = [] # model's answer list

    with open(json_path, 'r') as f:
        data = json.load(f)

    for item in data:
        value = str(item['value'])
        if value == "yes":
            label = 1
        elif value == "no":
            label = 0
        else:
            continue  # skip invalid labels

        answer = chat(model_name, item['prompt'])
        if answer == -1:
            continue  # skip invalid model outputs

        truth.append(label)
        modelAnswers.append(answer)

    print(f"Accuracy:  {accuracy_score(truth, modelAnswers):.2f}")
    print(f"Precision: {precision_score(truth, modelAnswers):.2f}")
    print(f"Recall:    {recall_score(truth, modelAnswers):.2f}")
