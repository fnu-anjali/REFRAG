#!/usr/bin/env python3
"""
Simple generation without model downloads - just formats the prompt
"""
import json
import sys

# Load JSON data
json_file = sys.argv[1] if len(sys.argv) > 1 else "data/run_0__insurance_regulatory__gpt-oss-20b.json"
question_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print(f"Loading {json_file}...")
with open(json_file, 'r') as f:
    data = json.load(f)

# Get question and context
item = data['data'][question_idx]
question = item['question']
contexts = item['context']

print(f"\nQuestion: {question}")
print(f"\nNumber of context chunks: {len(contexts)}")
print("\n" + "="*80)
print("FORMATTED PROMPT FOR LLM:")
print("="*80)

# Load and format prompt template
with open('prompt.txt', 'r') as f:
    prompt_content = f.read()

# Extract system and user prompts
import re
system_match = re.search(r"system = '''(.*?)'''", prompt_content, re.DOTALL)
user_match = re.search(r"user = '''(.*?)'''", prompt_content, re.DOTALL)

system_prompt = system_match.group(1).strip() if system_match else ""
user_template = user_match.group(1).strip() if user_match else ""

# Format user prompt
user_prompt = user_template.replace("{{ query }}", question)

# Format documents
docs_section = ""
for i, ctx in enumerate(contexts[:10], 1):  # Limit to first 10 for readability
    docs_section += f"({i}):\n  {ctx[:200]}...\n-----\n"

user_prompt = re.sub(
    r'\{% for doc in documents %\}.*?\{% endfor %\}',
    docs_section,
    user_prompt,
    flags=re.DOTALL
)

print("\n--- SYSTEM PROMPT ---")
print(system_prompt)
print("\n--- USER PROMPT ---")
print(user_prompt)

print("\n" + "="*80)
print(f"Ground Truth Answer:\n{item.get('ground_truth', 'N/A')}")
print("="*80)
