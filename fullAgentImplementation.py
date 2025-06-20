import json
import sys
import time
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score



docTypeClassifierSys = (
    "You are a preprocessing assistant that classifies the type and purpose of construction-related documents.\n\n"
    "Task: Read the first few chunks of the document and assign exactly one document type from the list below.\n\n"
    "Allowed types:\n"
    "• Request for Information\n"
    "• Change Order\n"
    "• Contract\n"
    "• Specification\n"
    "• Drawing\n"
    "• Schedule\n"
    "• Budget\n"
    "• Inspection Report\n"
    "• Permit\n"
    "• Other\n\n"
    "Guidelines:\n"
    "• Focus only on the initial chunks — identify the type using titles, headers, or key phrases.\n"
    "• Choose the most dominant or explicit category if multiple seem plausible.\n"
    "• If none of the above types match, return 'Other'.\n"
    "• Do not return reasoning or explanation.\n"
    "• Do not hallucinate or infer beyond what is written.\n"
    "• Your classification will be used to attach metadata to all document chunks downstream."
)

decompSys = """\
You are the decomposition module for a construction-contract Q&A pipeline.

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
keywordSys = """\
You are a helpful assistant. You will be given a user query that will be used as a search probe to retrieve information from contracts related to construction. 
Task: for each query, return a valid JSON that extracts 5 keywords from the query that will be the most helpful to search for during the retrieval process. 
If there are less than 5 keywords in the query, generate the remaining keyword(s) based on the fact that 
key words you are generating will be used to create subqueries that will be used as search probes for the retrieval process.
Output format(exactly one line, valid JSON, no extra text, no markdown fences):
{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","calculation"]}

"""
subquerySys = """\
You are a helpful assistant.  You will be given a user query that will be used as a search probe to retrieve information from contracts related to construction. You will also be given five associated keywords, that were either directly extracted from the query or was generated with the instruction of being a helpful keyword to search for during the retrieval process.
Task: decompose the original query into a sequence of subqueries that will be consumed by the answer synthesizer after the retrieval process, in order to guide the answer synthesizer to output an accurate response to the query.
Each subquery should:

- Contain at least one of the provided keywords.
- Represent a distinct piece of information the synthesizer needs to compose the final answer.
- Be phrased clearly and concisely, without redundancy or unnecessary detail.
- Together cover all aspects of the original query so the synthesizer can build a complete, accurate response.


Output format(no extra text, no markdown fences):
1. 'subquery 1' \n
2. 'subquery 2'\n
3. 'subquery 3'\n
... and so on until the subqueries cover the entire query. The fewshot you are given lists 4 subqueries but you should provide the correct amount of subqueries to cover the entire query, which may be more or less than 4.

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
fewshotKeyword = [
    {
        "role": "user",
        "content": "Who approves a change order over $50,000?"
    },
    {
        "role": "assistant",
        "content": '{"prompt":"Who approves a change order over $50,000?","keywords":["change order","approval","$50,000","authorization","threshold"]}'
    },
    {"role": "user", "content": "What are the liquidated damages?"},
    {
        "role": "assistant",
        "content": '{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","calculation"]}',
    },
]
fewshotSubquery = [
    {
        "role": "user",
        "content": '{ "prompt": "What are the liquidated damages?", "keywords": ["liquidated","damages","penalty","compensation","calculation"]}',
    },
    {
        "role": "assistant",
        "content": "1. 'Find the clause defining liquidated damages'\n2. 'Locate any penalty provisions related to damages'\n3. 'Extract compensation terms specified for damages'\n4. 'Extract numerical examples illustrating liquidated damages calculation'",
    },
    {
        "role": "user",
        "content": '{"prompt":"Define substantial completion in this contract.","keywords":["substantial completion","definition","completion date","certificate","milestone"]}'
    },
    {
        "role": "assistant",
        "content":
            "1. Locate the definition of substantial completion\n"
            "2. Extract any certificate or milestone language tied to completion date"
    },

    # Case needing 5 probes
    {
        "role": "user",
        "content": '{"prompt":"Who approves a change order over $50,000?","keywords":["change order","approval","$50,000","authorization","threshold"]}'
    },
    {
        "role": "assistant",
        "content":
            "1. Find clauses stating who grants approval for a change order\n"
            "2. Identify any dollar threshold for change order authorization\n"
            "3. Extract roles or titles responsible for $50,000 change order approval\n"
            "4. Retrieve procedural steps before change order authorization\n"
            "5. Locate signature or documentation requirements for change order approval"
    }
]


def chat(model: str, messages: list) -> str:
    return ollama.chat(
        model=model, messages=messages, options={"temperature": 0}, think=False
    )["message"]["content"].strip()


def decompose(model: str, prompt: str):
    msgs = [
        {"role": "system", "content": decompSys},
        *fewshotDecomp,
        {"role": "user", "content": prompt},
    ]
    raw = chat("qwen3:4b", msgs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_keywords(model: str, query: str) -> str:
    msgs = [
        {"role": "system", "content": keywordSys},
        *fewshotKeyword,
        {"role": "user", "content": query},
    ]
    return chat(model, msgs)


def plan_subqueries(model: str, kw_json: str) -> str:
    msgs = [
        {"role": "system", "content": subquerySys},
        *fewshotSubquery,
        {"role": "user", "content": kw_json},
    ]
    return chat(model, msgs)


def md_row(*cells):
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def print_per_prompt(prompt: str, decomp: dict):
    print(md_row(prompt, "context", "; ".join(decomp.get("context", []))))
    for q in decomp.get("queries", []):
        print(md_row(prompt, "query", q))
    for d in decomp.get("directives", []):
        print(md_row(prompt, "directive", d))
    for n in decomp.get("noise", []):
        print(md_row(prompt, "noise", n))


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 agent.py <ollama_model_name> <prompts.json>")
        sys.exit(1)

    model_name, path = argv
    with open(path) as fh:
        prompts_data = json.load(fh)

    truth, preds, decomp_json, kw_outputs, sub_outputs = [], [], [], [], []
    t0 = time.time()

    for item in prompts_data:
        prompt = item["prompt"]
        ground_truth = 1 if item.get("value") else 0

        decomp = decompose(model_name, prompt)
        decomp_json.append((prompt, decomp))

        is_query = 1 if decomp and decomp.get("queries") else 0
        preds.append(is_query)
        truth.append(ground_truth)

        # keyword + subquery for each individual rewritten query
        if decomp and decomp.get("queries"):
            for q in decomp["queries"]:
                kw_json = extract_keywords(model_name, q)
                subq = plan_subqueries(model_name, kw_json)
                kw_outputs.append((q, kw_json))
                sub_outputs.append((q, subq))

    elapsed = time.time() - t0
    if any(p in (0, 1) for p in preds):
        acc = accuracy_score(truth, preds)
        prec = precision_score(truth, preds, zero_division=0)
        rec = recall_score(truth, preds, zero_division=0)

        # --- Table 1: model metrics ---
        # print("\n| Model | Accuracy | Precision | Recall | Total Time (s) | Avg Time per Prompt (s) |")
        # print("|---|---|---|---|---|---|")
        # print(f"| {model_name} | {acc:.2f} | {prec:.2f} | {rec:.2f} | {elapsed:.2f} | {elapsed/len(truth):.2f} |")

        # --- Table 2: per prompt ---
        # print("\n| Prompt | Ground Truth | Decomposer Output |")
        # print("|---|---|---|")
        # for (prompt, dec), pred, gt in zip(decomp_json, preds, truth):
        #     display = json.dumps(dec, separators=(",", ":")) if dec else "Invalid JSON"
        #     p = prompt.replace("|", "\\|")
        #     print(f"| {p} | {'true' if gt else 'false'} | {display} |")
        print("\n### Prompt Categories\n")
        print("| Prompt | Category |")
        print("|---|---|")
        for prompt, dec in decomp_json:
            # decide category
            cat = "Query" if dec and dec.get("queries") else "Non-Query"
            # escape pipes in the prompt
            p = prompt.replace("|", "\\|")
            print(f"| {p} | {cat} |")

        if kw_outputs:
            print("\n| Query | Keywords | Subqueries |")
            print("|---|---|---|")
            for (query, kw_raw), (_, subqs) in zip(kw_outputs, sub_outputs):
                # 1) Parse the keywords JSON
                try:
                    kw_obj = json.loads(kw_raw)
                    kw_list = kw_obj.get("keywords", [])
                except json.JSONDecodeError:
                    kw_list = [kw_raw]  # fallback in case of parse error

                # 2) Clean up each cell
                clean_q = query.replace("|", "\\|").strip()
                clean_kw = ", ".join(kw_list).replace("|", "\\|")
                # Flatten subqueries into one cell (semicolon-separated)
                clean_subqs = " ; ".join(line.strip() for line in subqs.splitlines())
                clean_subqs = clean_subqs.replace("|", "\\|").replace("'", "")

                # 3) Print the row
                print(f"| {clean_q} | {clean_kw} | {clean_subqs} |")
        else:
            print("No valid predictions to score.")


if __name__ == "__main__":
    main(sys.argv[1:])
