import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------
# 1) Question Function
# -------------------------------------------------------
def ask(title, options):
    print(f"\n=== {title} ===")
    for i, opt in enumerate(options, start=1):
        print(f"[{i}] {opt}")

    while True:
        try:
            choice = int(input("Choose number: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


# -------------------------------------------------------
# 2) Choose Parameters
# -------------------------------------------------------

DATASET = ask(
    "Available datasets", 
    ['ml-1m', 'steam', 'goodreads']
)


prompt_files = sorted([
    f for f in os.listdir(os.path.join(BASE_DIR, "prompts"))
    if f.endswith(".txt")
])
screened_prompt_files = [
    f for f in prompt_files
    if f.split('_')[1] == DATASET
]
if len(screened_prompt_files) == 0:
    raise RuntimeError(
        f"No prompt found for dataset '{DATASET}' in prompts/"
    )
if len(screened_prompt_files) > 1:
    raise RuntimeError(
        f"Multiple prompts found for dataset '{DATASET}': {screened_prompt_files}"
    )
PROMPT_FILE = screened_prompt_files[0]


dry_run_choice = ask(
    "Execution mode",
    ["Dry run (print command only)", "Launch with nohup"]
)
DRY_RUN = dry_run_choice.startswith("Dry")


# -------------------------------------------------------
# 3) Paths & checks
# -------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
base_artifacts = os.path.join(PROJECT_ROOT, "CausalI2I_artifacts")

chosen_pairs_path = os.path.join(
    base_artifacts,
    "Chosen_Pairs",
    f"{DATASET}_chosen_pairs.pkl"
)

prompt_path = os.path.join(
    BASE_DIR, "prompts", PROMPT_FILE
)

results_dir = os.path.join(
    base_artifacts, 
    "API_Results", 
    DATASET
)

if not os.path.exists(chosen_pairs_path):
    print(f"\nChosen pairs file not found: {chosen_pairs_path}")
    raise SystemExit(0)
if not os.path.exists(prompt_path):
    print(f"\nPrompt file not found: {prompt_path}")
    raise SystemExit(0)

os.makedirs(results_dir, exist_ok=True)


# -------------------------------------------------------
# 4) Command construction
# -------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(results_dir, f"log_{timestamp}.txt")

cmd = (
    f"nohup python3 -u run_GPT.py "
    f"--dataset {DATASET} "
    f"--prompt {PROMPT_FILE} "
    f"> '{log_file}' 2>&1 &"
)


# -------------------------------------------------------
# 5) Final confirmation
# -------------------------------------------------------
print("\n=== Final configuration ===")
print(f"Dataset       : {DATASET}")
print(f"Prompt        : {PROMPT_FILE}")
print(f"Chosen pairs  : {chosen_pairs_path}")
print(f"Results dir   : {results_dir}")
print(f"Log file      : {log_file}")
print(f"Dry run       : {DRY_RUN}")

print("\nCommand:")
print(cmd)

confirm = ask("Confirm launch", ["Cancel", "Launch"])

if confirm == "Cancel":
    print("Aborted.")
    raise SystemExit(0)

# -------------------------------------------------------
# 6) Execute
# -------------------------------------------------------
if DRY_RUN:
    print("\n🧪 Dry run only — command not executed.")
else:
    os.system(cmd)
    print("\n🚀 Experiment launched with nohup.")


# ps aux | grep run_GPT.py
# kill 123456
