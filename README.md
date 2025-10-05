## PolyglotQL: Multilingual Text-to-SPARQL

This repository has the code for:
1. Collecting & cleaning QALD-style question–SPARQL pairs.
2. Fine-tuning 4-bit LoRA adapters for causal LMs (Mistral / Occiglot) on text→SPARQL.
3. Executing & scoring generated SPARQL over Wikidata (precision / recall / F1 + error categories).

Contains two versions for the training code (`training/v1` experiments; `training/v2` pipeline), validation utilities, and QALD harvesting scripts.

---
### Quick Start
Create env & install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r training/requirements.txt
pip install -r validation/requirements.txt
```
Optional Hugging Face login:
```bash
export HF_TOKEN=hf_xxx
```

---
### 1. Collect QALD Data
`data/collection/QALD_dataset_generator/main.py` aggregates multilingual QALD sources and (optionally) pushes to a HF dataset (edit `HUGGING_FACE_REPO`).
```bash
cd data/collection/QALD_dataset_generator
python main.py
```
Outputs: `qald_challenges.csv` (deduped & filtered).

---
### 2. Prepare Train/Eval JSON
Create JSONL for training (example schema):
```json
{"instruction": "Question text ...", "output": "SELECT ..."}
```
Save e.g. `data/v2/train.json`, `data/v2/eval.json`.

---
### 3. Fine-tune (LoRA 4-bit) – v2
Entrypoint: `training/v2/finetune.py` (wraps TRL SFT). Key arg groups: model, data, LoRA, training, custom early stopping.
Example:
```bash
python training/v2/finetune.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --train_file data/v2/train.json \
  --eval_file data/v2/eval.json \
  --output_dir outputs/mistral_en_v2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lora_alpha 16 --r 64 --lora_dropout 0.01
```
Result: adapter + tokenizer in `outputs/mistral_en_v2` (best checkpoint; early stopping on `bleu`).

---
### 4. Validate SPARQL
`validation/sparql_validator.py` executes gold vs generated queries (Wikidata endpoint; caching + retries) and assigns categories (execution failure, empty, exact, partial, no overlap) plus P/R/F1.
Expected keys: `question`, `generated_sparql`, `gold_sparql`.

---
### Structure (essentials)
```
data/collection/QALD_dataset_generator/  # QALD merge/clean
training/v1/                             # Earlier scripts
training/v2/                             # Argument-driven fine-tuning
validation/                              # SPARQL metrics & plots
plotting_scripts/                        # Loss / grad norm plots
```

### Notes
- 4-bit (nf4) + LoRA (`BitsAndBytesConfig`, `peft`).
- Auto pad token & chat formatting (`trl.setup_chat_format`).
- Early stopping patience via `--early_stopping_patience`.
- Metric placeholder = BLEU
