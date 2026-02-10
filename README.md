# CausalI2I

This repository contains the code and experiments for the CausalI2I project.

**Important: required artifacts folder**  
This project looks for a sibling directory named `CausalI2I_artifacts` **next to** this repo. It stores datasets, intermediate outputs, and other assets used by the scripts and notebooks. If it is missing, just ask the project owner for it or restore it from your local backup.

Example layout:
```
Home directory
  ├── CausalI2I
  └── CausalI2I_artifacts
```

**Prereqs**  
Run notebooks from their own folders (the code uses `Path.cwd()` and `parents[...]` to locate `CausalI2I_artifacts`).  
You will need a Python environment with typical data-science deps: `numpy`, `pandas`, `torch`, `tqdm`, `matplotlib`, `scikit-learn`, `scipy`, and `openai` for the GPT step.  
For the GPT step, place your OpenAI API key in `~/secret_api_key.txt`.

**Run Order (1 → 7)**  
*The `CausalI2I_artifacts` folder already contains results from each step, so you can skip steps if you want. Run everything if you want to reproduce all results.* 
Go in order from folder `1_...` to `7_...`. Each step depends on outputs from previous steps.

1. **1_Preprocessing**  
`1_Preprocessing/dataset_processors/processor_ml-1m.ipynb` loads `CausalI2I_artifacts/Datasets/Raw/ml-1m/ratings.dat`, builds train/test splits, reindexes users/items, and writes `CausalI2I_artifacts/Datasets/Processed/ml-1m/train.csv`, `test.csv`, `data_sasrec.csv`, plus `item_dict.pkl` and `Chosen_Pairs/ml-1m_chosen_pairs.pkl`.  
`1_Preprocessing/dataset_processors/processor_steam.ipynb` does the same for `CausalI2I_artifacts/Datasets/Raw/steam/steam_filtered.csv` and writes the processed Steam files.  
`1_Preprocessing/dataset_processors/processor_goodreads.ipynb` does the same for `CausalI2I_artifacts/Datasets/Raw/goodreads/goodreads_filtered.csv`, with extra duplicate handling, and writes the processed Goodreads files.  
`1_Preprocessing/descriptions/describer.ipynb` reads the processed `train.csv`/`test.csv` and writes a dataset summary text file like `1_Preprocessing/descriptions/<DATASET>_description.txt`.  
Pick the dataset(s) you want and run the matching processor notebook(s). These outputs are required by every later step.

2. **2_Propensities**  
`2_Propensities/train_MF.ipynb` trains a Matrix Factorization model on `CausalI2I_artifacts/Datasets/Processed/<DATASET>/train.csv` (with validation on `test.csv`) and saves the propensity model to `CausalI2I_artifacts/Propensity_Models/MF<n_factors>_<DATASET>.pt`.  
This model provides propensity scores used in later evaluation.

3. **3_ChatGPT**  
*Note: You do not have to run this step if `CausalI2I_artifacts` already contains GPT results from previous runs.* 
`3_ChatGPT/launch_GPT.py` is the recommended entry point. It asks you to choose the dataset and prompt, validates `Chosen_Pairs/<DATASET>_chosen_pairs.pkl`, and builds a `nohup` command for `run_GPT.py`.  
`3_ChatGPT/run_GPT.py` sends the chosen title pairs to the OpenAI API (model `gpt-5.2`) in batches and writes results to `CausalI2I_artifacts/API_Results/<DATASET>/causal_scores_final_YYYY-MM-DD.csv` (plus a partial file during the run).  
You must have `~/secret_api_key.txt` with your API key, and the `prompts/` files must match the dataset you select.

4. **4_SASRec**  
`4_SASRec/SASRec_train.ipynb` loads `CausalI2I_artifacts/Datasets/Processed/<DATASET>/data_sasrec.csv`, trains a SASRec model, and saves `CausalI2I_artifacts/SASRec_Models/sasrec_<DATASET>.pt` plus `sasrec_<DATASET>_init_dict.pkl`.  
These files are required for evaluation.

5. **5_Evaluation**  
`5_Evaluation/calculate_metrics.ipynb` loads the processed dataset, the propensity model, GPT results from `API_Results`, and the SASRec model. It computes ATE/STD and baseline scores and writes `CausalI2I_artifacts/Datasets/Evaluated/<DATASET>_evaluated.csv`.  
`5_Evaluation/comparison.ipynb` reads that evaluated file and generates figures in `CausalI2I_artifacts/Figures/<DATASET>/`, including `ate_ste_vs_causal_effect.png`, `precision_recall_at_k.png`, `pr_roc_curves.png`, and `metric_distribution_by_causal_effect.png`.

6. **6_Sequels**  
`6_Sequels/6.1_train_test_split.ipynb` builds a sequel-only dataset from Goodreads using `CausalI2I_artifacts/Datasets/Sequels/name2series.pkl` and `Datasets/Processed/goodreads/*`, then writes `CausalI2I_artifacts/Datasets/Sequels/train.csv`, `test.csv`, and `id2info.pkl`.  
`6_Sequels/6.2_train_MF.ipynb` trains an MF model on the sequels dataset and saves `CausalI2I_artifacts/Propensity_Models/MF25_sequels.pt`.  
`6_Sequels/6.3_calculate_metrics.ipynb` evaluates causal metrics for sequels using the MF model and the Goodreads SASRec model and writes `CausalI2I_artifacts/Datasets/Sequels/sequels_evaluated.csv`.  
`6_Sequels/6.4_comparison.ipynb` generates the sequels figure `CausalI2I_artifacts/Figures/sequels/Binned Precision.png`.

7. **7_Simulation**  
`7_Simulation/7.1_train_test_split.ipynb` loads `CausalI2I_artifacts/Datasets/Simulation/synthetic.csv` and `ground_truth.csv`, builds train/test splits, and writes `train.csv`, `test.csv`, and `ground_truth_processed.csv`.  
`7_Simulation/7.2_train_MF.ipynb` trains an MF model on the simulation data and saves `CausalI2I_artifacts/Propensity_Models/MF20_simulation.pt`.  
`7_Simulation/7.3_calculate_metrics.ipynb` evaluates causal metrics against the processed ground truth and writes `CausalI2I_artifacts/Datasets/Simulation/simulation_evaluated.csv`.  
`7_Simulation/7.4_comparison.ipynb` creates simulation figures in `CausalI2I_artifacts/Figures/simulation/`, including `precision_recall_synthetic.png`, `pr_roc_synthetic.png`, `distribution_bins_synthetic.png`, and `scatter_ate_vs_ste_causal_effect_(linked_only).png`.
