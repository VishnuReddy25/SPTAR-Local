# Detailed Explanation and Execution Order of Retrieval Commands

This document provides a detailed explanation of the commands used for retrieval experiments on the `fiqa` and `msmarco` datasets using BM25, DPR, ColBERT, and BM25CE methods. It also provides the recommended order of execution for these commands to help new users run the experiments smoothly. Duplicates have been removed for efficiency.

---

## Prerequisites

1. Pull the Docker image once for Pyserini FastAPI server (used in BM25 and BM25CE):
   ```
   docker pull beir/pyserini-fastapi
   ```
2. Ensure Python 3.7 environment is available for ColBERT data generation.
3. Set up the environment with required packages.

---

## BM25 (Lexical Retrieval)

BM25 is a classical lexical retrieval method based on term frequency and inverse document frequency. The implementation uses Anserini/Pyserini with a Docker container for indexing and searching.

### Commands for fiqa dataset

1. Run the Docker container exposing port 8002 for fiqa:
   ```
   docker run -p 8002:8000 -it --name fiqa --rm beir/pyserini-fastapi:latest
   ```
2. Run the BM25 evaluation script:
   ```
   python zhiyuan/retriever/bm25anserini/evaluate_anserini_bm25.py --dataset_name fiqa
   ```

### Commands for msmarco dataset

1. Run the Docker container exposing port 8000 for msmarco:
   ```
   docker run -p 8000:8000 -it --name msmarco --rm beir/pyserini-fastapi:latest
   ```
2. Run the BM25 evaluation script:
   ```
   python zhiyuan/retriever/bm25anserini/evaluate_anserini_bm25.py --dataset_name msmarco
   ```

---

## W/O Variant (No Augmentation)

### DPR (Dense Passage Retrieval)

DPR uses dense vector representations for queries and documents. The commands below train and evaluate DPR models without any data augmentation.

- For fiqa:
  ```
  python zhiyuan/dpr_eval.py --dataset_name fiqa --version v1 --gpu_id 0 --train_num 50 -exps no_aug --weak_num 100k
  ```
- For msmarco:
  ```
  python zhiyuan/dpr_eval.py --dataset_name msmarco --version v1 --gpu_id 0 --train_num 50 -exps no_aug --weak_num 100k
  ```
- Testing results are logged in:
  ```
  zhiyuan/retriever/dpr/train/output/no_aug/
  ```

### ColBERT (Late Interaction)

ColBERT is a late interaction model that balances efficiency and effectiveness.

- For fiqa:
  1. Generate ColBERT training data (run in Python 3.7 environment):
     ```
     python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name fiqa --exp_name no_aug
     ```
  2. Train ColBERT model:
     ```
     bash zhiyuan/retriever/col_bert/train_colbert.sh -g 0,1,2,3 -d fiqa -e no_aug -m 80 -s 4 -b 128
     ```
  3. Test ColBERT model:
     ```
     bash zhiyuan/retriever/col_bert/test_colbert.sh -g 0,1,2,3 -d fiqa -e no_aug -p 96 -c 80
     ```

- For msmarco:
  1. Generate ColBERT training data:
     ```
     python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name msmarco --exp_name no_aug
     ```
  2. Train ColBERT model:
     ```
     bash zhiyuan/retriever/col_bert/train_colbert.sh -g 0,1,2,3 -d msmarco -e no_aug -m 40 -s 2 -b 128
     ```
  3. Test ColBERT model:
     ```
     bash zhiyuan/retriever/col_bert/test_colbert.sh -g 0,1,2,3 -d msmarco -e no_aug -p 2000 -c 40
     ```
- Testing results are documented in `$LOG_DIR/test_log.txt` where `LOG_DIR` is defined in `zhiyuan/retriever/col_bert/test_colbert.sh`.

---

## BM25CE (BM25 + Cross-Encoder Reranking)

BM25CE first retrieves documents using BM25 and then reranks them using a cross-encoder model.

### Commands for fiqa

1. Run Docker container:
   ```
   docker run -p 8002:8000 -it --name fiqa --rm beir/pyserini-fastapi:latest
   ```
2. Run BM25CE evaluation:
   ```
   python zhiyuan/retriever/bm25ce/eval/evaluate_bm25_ce_dpr.py --dataset_name fiqa --exp_name no_aug --topk 1000
   ```

### Commands for msmarco

1. Run Docker container:
   ```
   docker run -p 8000:8000 -it --name msmarco --rm beir/pyserini-fastapi:latest
   ```
2. Run BM25CE evaluation:
   ```
   python zhiyuan/retriever/bm25ce/eval/evaluate_bm25_ce_dpr.py --dataset_name msmarco --exp_name no_aug --topk 1000
   ```
- Testing results are logged in:
  ```
  zhiyuan/retriever/bm25ce/eval/output/no_aug
  ```

---

## InPars Variant (Instruction-based Parsing Augmentation)

### DPR

- For fiqa:
  ```
  python zhiyuan/dpr_eval.py --dataset_name fiqa --version v1 --gpu_id 0 --train_num 50 -exps p_written_100k_vicuna_prompt_2_filtered_70 --weak_num 100k
  ```
- For msmarco:
  ```
  python zhiyuan/dpr_eval.py --dataset_name msmarco --version v1 --gpu_id 0 --train_num 50 -exps p_written_100k_vicuna_prompt_3_filtered_30 --weak_num 100k
  ```

### ColBERT

- For fiqa:
  1. Generate ColBERT training data (run in Python 3.7 environment):
     ```
     python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name fiqa --exp_name p_written_100k_vicuna_prompt_2_filtered_70
     ```
  2. Train ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/train_colbert.sh -g 0,1,2,3 -d fiqa -e p_written_100k_vicuna_prompt_2_filtered_70 -m 1200 -s 60 -b 128
     ```
  3. Test ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/test_colbert.sh -g 0,1,2,3 -d fiqa -e p_written_100k_vicuna_prompt_2_filtered_70 -p 96 -c 120
     ```

- For msmarco:
  1. Generate ColBERT training data:
     ```
     python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name msmarco --exp_name p_written_100k_vicuna_prompt_3_filtered_30
     ```
  2. Train ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/train_colbert.sh -g 0,1,2,3 -d msmarco -e p_written_100k_vicuna_prompt_3_filtered_30 -m 6300 -s 315 -b 128
     ```
  3. Test ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/test_colbert.sh -g 0,1,2,3 -d msmarco -e p_written_100k_vicuna_prompt_3_filtered_30 -p 2000 -c 6300
     ```

### BM25CE

- For fiqa:
  ```
  docker run -p 8002:8000 -it --name fiqa --rm beir/pyserini-fastapi:latest
  python zhiyuan/retriever/bm25ce/eval/evaluate_bm25_ce_dpr.py --dataset_name fiqa --exp_name p_written_100k_vicuna_prompt_2_filtered_70 --topk 1000
  ```

- For msmarco:
  ```
  docker run -p 8000:8000 -it --name msmarco --rm beir/pyserini-fastapi:latest
  python zhiyuan/retriever/bm25ce/eval/evaluate_bm25_ce_dpr.py --dataset_name msmarco --exp_name p_written_100k_vicuna_prompt_3_filtered_30 --topk 1000
  ```

---

## SPTAR Variant (Soft Prompt Tuning Augmentation)

### DPR

- For fiqa:
  ```
  python zhiyuan/dpr_eval.py --dataset_name fiqa --version v1 --gpu_id 0 --train_num 50 -exps llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 --weak_num 100k
  ```

- For msmarco:
  ```
  python zhiyuan/dpr_eval.py --dataset_name msmarco --version v1 --gpu_id 0 --train_num 50 -exps llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30 --weak_num 100k
  ```

### ColBERT

- For fiqa:
  1. Generate ColBERT data (run in Python 3.7 environment):
     ```
     python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name fiqa --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
     ```
  2. Train ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/train_colbert.sh -g 0,1,2,3 -d fiqa -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 3900 -s 195 -b 128
     ```
  3. Test ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/test_colbert.sh -g 0,1,2,3 -d fiqa -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -p 96 -c 975
     ```

- For msmarco:
  1. Generate ColBERT data:
     ```
     python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name msmarco --exp_name llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30
     ```
  2. Train ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/train_colbert.sh -g 0,1,2,3 -d msmarco -e llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30 -m 6460 -s 323 -b 128
     ```
  3. Test ColBERT:
     ```
     bash zhiyuan/retriever/col_bert/test_colbert.sh -g 0,1,2,3 -d msmarco -e llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30 -p 2000 -c 6460
     ```

### BM25CE

- For fiqa:
  ```
  docker run -p 8002:8000 -it --name fiqa --rm beir/pyserini-fastapi:latest
  python zhiyuan/retriever/bm25ce/eval/evaluate_bm25_ce_dpr.py --dataset_name fiqa --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 --topk 1000
  ```

- For msmarco:
  ```
  docker run -p 8000:8000 -it --name msmarco --rm beir/pyserini-fastapi:latest
  python zhiyuan/retriever/bm25ce/eval/evaluate_bm25_ce_dpr.py --dataset_name msmarco --exp_name llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30 --topk 1000
  ```

---

## Weak Data Filter Module

This module filters weak document-query pairs using BM25 scores.

- Example: `llama_7b_100k_fixed_v4_best_llama_prompt_3_filtered_30` comes from filtering `llama_7b_100k_fixed_v4_best_llama_prompt_3` by `zhiyuan/filter/bm25anserini_split.py` with topk=30, where `llama_7b_100k_fixed_v4_best_llama_prompt_3` contains raw 100k weak document-query pairs generated by the soft prompt augmentor module.

---

## Full Project Execution Order: From py37 Creation to Dense Retrieval Training

This section provides the complete order of commands to execute the entire project from creating the Python 3.7 environment to training dense retrieval models. It includes what happens with each command.

### Step 1: Create Python 3.7 Environment
- Command: `conda create -n py37 python=3.7`
- What happens: Creates a new Conda environment named 'py37' with Python 3.7, which is required for ColBERT data generation and training.

### Step 2: Activate Environment and Install Dependencies
- Command: `conda activate py37 && pip install -r requirements.txt`
- What happens: Activates the py37 environment and installs all required Python packages from requirements.txt, setting up the dependencies for the project.

### Step 3: Pull Docker Image for BM25/BM25CE
- Command: `docker pull beir/pyserini-fastapi`
- What happens: Downloads the Docker image for Pyserini FastAPI server, which is used for BM25 indexing and retrieval in BM25 and BM25CE methods.

### Step 4: Generate and Filter Weak Data (for Augmented Variants)
- Command: `python zhiyuan/filter/bm25anserini_split.py --input_file raw_pairs.json --topk 30 --output_file filtered_pairs.json`
- What happens: Filters weak document-query pairs using BM25 scores, keeping only the top 30 relevant pairs per query to improve training data quality for InPars/SPTAR variants.

### Step 5: Train DPR Model
- Command: `python zhiyuan/dpr_eval.py --dataset_name {dataset} --version v1 --gpu_id 0 --train_num 50 -exps {variant} --weak_num 100k`
- What happens: Trains a Dense Passage Retrieval (DPR) model using the specified dataset and variant, encoding queries and passages into dense vectors for retrieval. The model learns to match relevant passages to queries.

### Step 6: Evaluate DPR Model
- What happens: The same command evaluates the trained DPR model on the test set, computing metrics like NDCG, MAP, and Recall. Results are logged in `zhiyuan/retriever/dpr/train/output/{variant}/`.

### Step 7: Generate ColBERT Training Data
- Command: `python zhiyuan/retriever/dpr/train/gen_data_for_colbert.py --dataset_name {dataset} --exp_name {variant}`
- What happens: Prepares training data for ColBERT by sampling negative documents for each query (using the same data as DPR for consistency), formatting it for ColBERT's late interaction training.

### Step 8: Train ColBERT Model
- Command: `bash zhiyuan/retriever/col_bert/train_colbert.sh -g 0,1,2,3 -d {dataset} -e {variant} -m {steps} -s {save_steps} -b 128`
- What happens: Trains the ColBERT model using distributed GPUs, learning token-level interactions between queries and passages for efficient retrieval.

### Step 9: Test ColBERT Model
- Command: `bash zhiyuan/retriever/col_bert/test_colbert.sh -g 0,1,2,3 -d {dataset} -e {variant} -p {partitions} -c {checkpoint}`
- What happens: Indexes the corpus, retrieves relevant passages, and evaluates the ColBERT model on the test set. Results are logged in `$LOG_DIR/test_log.txt`.

### Step 10: Run BM25CE Evaluation
- Command: `docker run -p 800{port}:8000 -it --name {dataset} --rm beir/pyserini-fastapi:latest && python zhiyuan/retriever/bm25ce/eval/evaluate_bm25_ce_dpr.py --dataset_name {dataset} --exp_name {variant} --topk 1000`
- What happens: Starts a Docker container for BM25 retrieval, then reranks the top 1000 retrieved documents using a cross-encoder model trained on DPR. Results are logged in `zhiyuan/retriever/bm25ce/eval/output/{variant}`.

### Step 11: Monitor and Clean Up
- What happens: Check log files for metrics, stop Docker containers with `docker stop {dataset}`, and deactivate the environment with `conda deactivate`.

---

## Notes on Execution Order

1. For each variant and dataset, start with BM25 evaluation if you want a lexical baseline.
2. For DPR, run the training and evaluation command for the chosen variant.
3. For ColBERT, first generate the training data (in Python 3.7 environment), then train, then test.
4. For BM25CE, run the Docker container, then run the evaluation script.
5. For augmented variants (InPars, SPTAR), ensure weak data filtering is done before training.
6. Monitor logs and outputs in the specified directories for evaluation metrics.
7. Stop Docker containers after BM25/BM25CE runs to free resources.

---

This detailed explanation and execution order should help new users understand and run the retrieval experiments effectively.
