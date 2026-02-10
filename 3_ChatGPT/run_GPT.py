
import pandas as pd
import json
import time
import os
import pickle
import argparse
from datetime import datetime
import time
from openai import OpenAI

# -------------------------------------------------------
# 1) Set up paths and parameters
# -------------------------------------------------------

def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

start_time = get_now()
print(f'{start_time} - Session started.')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
base_artifacts = os.path.join(PROJECT_ROOT, "CausalI2I_artifacts")

from pathlib import Path

KEY_PATH = Path.home() / "secret_api_key.txt"

if not KEY_PATH.exists():
    raise RuntimeError(
        "OpenAI API key not found at ~/secret_api_key.txt"
    )

api_key = KEY_PATH.read_text().strip()
client = OpenAI(api_key=api_key)

# -------------------------------------------------------
# 2) Ask for dataset and prompt
# -------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--prompt", required=True, type=str)
args = parser.parse_args()

DATASET = args.dataset
PROMPT_FILE = args.prompt
print(f"Dataset: {DATASET}")
print(f"Prompt: {PROMPT_FILE}")

# -------------------------------------------------------
# 3) Load chosen title pairs and prompt
# -------------------------------------------------------

chosen_pairs_path = os.path.join(
    base_artifacts,
    "Chosen_Pairs",
    f"{DATASET}_chosen_pairs.pkl"
)

if not os.path.exists(chosen_pairs_path):
    raise RuntimeError(f"Chosen pairs file not found: {chosen_pairs_path}")

with open(chosen_pairs_path, "rb") as f:
    title_pairs = pickle.load(f)

# Load prompt
prompt_path = os.path.join(BASE_DIR, "prompts", PROMPT_FILE)

if not os.path.exists(prompt_path):
    raise RuntimeError(f"Prompt file not found: {prompt_path}")

with open(prompt_path, "r", encoding="utf-8") as f:
    PROMPT = f.read()

# Create results directory
results_dir = os.path.join(
    base_artifacts,
    "API_Results",
    DATASET
)
os.makedirs(results_dir, exist_ok=True)
partial_path = os.path.join(results_dir, "causal_scores_partial.csv")
final_path   = os.path.join(results_dir, f"causal_scores_final_{get_now()[:10]}.csv")

# -------------------------------------------------------
# 4) Test API connection
# -------------------------------------------------------

response = client.responses.create(
    model="gpt-5.2",
    input="Answer only \'Connected to ChatGPT API successfully.\'"
)

print(f'{get_now()} - {response.output_text}')

# -------------------------------------------------------
# 5) Process title pairs in batches and save results
# -------------------------------------------------------
def get_movies_causal_score(chunk, prompt, model="gpt-5.2"):
    history = ([
        {"role":"user", "content":prompt}
    ] + [
        {"role":"user", "content":f"A: '{title_A}'; B: '{title_B}'"}    
        for title_A, title_B in chunk
    ])
    response = client.responses.create(
        model=model,
        input=history,
        temperature=0,
        top_p=1
    )

    return response.output_text.strip()

batch_size = 10
batches = [title_pairs[i:i + batch_size] for i in range(0, len(title_pairs), batch_size)]

results = []
for i, batch in enumerate(batches, start=1):
    # print(f"⏳ Processing batch {batch_i} ({len(pairs_chunk)} pairs)...")

    success = False
    for attempt in range(3):
        response_text = get_movies_causal_score(batch, PROMPT)
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ Batch {i}, attempt {attempt+1}, JSON error: {e}")
            continue

        if len(data) != len(batch):
            print(
                f"⚠️ Batch {i}, attempt {attempt+1}: "
                f"expected {len(batch)} pairs in response, got only {len(data)}."
            )
            continue
        
        # only append once invariant is satisfied
        results.extend(data)
        success = True
        break

    if not success:
        print(f"Batch {i} failed after 3 attempts.")

    df_partial = pd.DataFrame(results)
    df_partial.to_csv(partial_path, index=False)
    
    # Report progress after every 10 batches
    if i % 10 == 0:
        print(f"{get_now()} - total pairs processed: {len(results)}")

    # Brief pause to avoid rate limits
    time.sleep(0.1)  


df_final = pd.DataFrame(results)
df_final.to_csv(final_path, index=False)
if os.path.exists(partial_path):
    os.remove(partial_path)

# -------------------------------------------------------
# 6) Report session duration
# -------------------------------------------------------
end_time = get_now()
print(f'{end_time} - Session completed.')

time_delta = pd.to_datetime(end_time) - pd.to_datetime(start_time)
elapsed_str = time.strftime(
    "%H:%M:%S",
    time.gmtime(time_delta.total_seconds())
)

print(f'\nTotal Duration: {elapsed_str}')
