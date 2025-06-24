import ollama
from outlines import from_ollama, Generator
from outlines.types import JsonSchema

# Step 1: Connect to Ollama
client = ollama.Client()
base_model = from_ollama(client, "llama3.2")

# Step 2: Define schema as a string
schema_str = """
{
  "type": "object",
  "properties": {
    "context": { "type": "array", "items": { "type": "string" } },
    "queries": { "type": "array", "items": { "type": "string" } },
    "directives": { "type": "array", "items": { "type": "string" } },
    "noise": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["context", "queries", "directives", "noise"],
  "additionalProperties": false
}
"""

# Step 3: Create the structured generator
generator = Generator(base_model, JsonSchema(schema_str))

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

user_prompt = "Hi! Can you tell me what clauses apply to late delivery? Also, please format your answer as a table."
full_prompt = decompSys + "\n\n" + user_prompt

# Step 4: Generate the structured output
output = generator(full_prompt)
print(output)
