# SPTAR Project: A Comprehensive Guide to Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models

## Table of Contents

1. Introduction to SPTAR
2. Project Structure and Overview
3. Data Preparation and Processing
4. Soft Prompt Fine-Tuning
5. Weak Query Generation (Augmentation)
6. Weak Data Filtering
7. Dense Retriever Training
8. Loss Functions and Training Details
9. Key Files and Their Roles
10. Running the Project: Step-by-Step Guide
11. Advanced Topics and Extensions
12. Conclusion

---

## Chapter 1: Introduction to SPTAR

The SPTAR (Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models) project is an implementation of a novel framework designed to enhance dense retrieval (DR) systems by leveraging large language models (LLMs) for data augmentation. Traditional dense retrieval models, such as DPR (Dense Passage Retrieval), suffer from limited training data, leading to suboptimal performance. SPTAR addresses this by using soft prompt tuning to generate high-quality weak queries for unlabeled documents, thereby augmenting the training data without requiring extensive manual labeling.

### Key Innovations

- **Soft Prompt Tuning**: Instead of fine-tuning the entire LLM, only a small set of trainable embeddings (soft prompts) are optimized, making the process efficient and parameter-efficient.
- **Weak Data Augmentation**: Generates synthetic query-document pairs from unlabeled corpora.
- **Filtering Mechanism**: Uses BM25-based retrieval to filter out noisy weak pairs.
- **Integration with Dense Retrieval**: Trains DR models on the augmented dataset, improving retrieval accuracy.

### Motivation

Dense retrieval is crucial for tasks like open-domain question answering and information retrieval. However, training effective DR models requires large amounts of labeled data (query-document pairs). SPTAR reduces this dependency by synthesizing weak labels using LLMs, achieving state-of-the-art performance on benchmarks like MS MARCO and FiQA.

---

## Chapter 2: Project Structure and Overview

The repository is organized into several directories and files, each serving a specific purpose in the SPTAR pipeline.

### Main Directories

- **xuyang/**: Handles LLM-based components, including soft prompt tuning, filtering, and augmentation.
- **zhiyuan/**: Manages data processing, weak data filtering, and dense retrieval training/evaluation.
- **package/**: Contains modified dependencies, such as BEIR and SentenceTransformers, for reproducibility.
- **imgs/**: Workflow diagrams and visualizations.

### Core Workflow

1. **Data Preparation**: Download and preprocess datasets.
2. **Soft Prompt Tuning**: Fine-tune soft prompts on labeled data.
3. **Augmentation**: Generate weak queries for unlabeled documents.
4. **Filtering**: Clean weak data using retrieval-based filtering.
5. **Training**: Train dense retrievers on augmented data.
6. **Evaluation**: Assess performance on test sets.

### Dependencies

- Python 3.7+ (for DPR, BM25CE)
- ColBERT environment (for ColBERT-based retrievers)
- Libraries: Transformers, PEFT, SentenceTransformers, BEIR, Pyserini (for BM25).

---

## Chapter 3: Data Preparation and Processing

Data preparation is the foundation of the SPTAR pipeline. The project uses standard IR datasets from BEIR (Benchmarking Information Retrieval).

### Datasets Used

- **MS MARCO**: Large-scale passage ranking dataset.
- **FiQA**: Financial question answering.
- **DL2019/DL2020**: TREC Deep Learning tracks.
- **Others**: Scifact, Quora, etc.

### Key Steps in Data Processing (`zhiyuan/data_process.py`)

1. **Download Data**: Use `zhiyuan/download.py` to fetch datasets from BEIR.
2. **Filter Unlabeled Corpus**: Remove documents that are already labeled to create an unlabeled pool.
   - Function: `filter_unlabeled_corpus()`
   - Input: Raw corpus.jsonl, train/dev/test.tsv
   - Output: corpus_filtered.jsonl
3. **Sample Corpus**: Create reduced subsets for efficient training.
   - Function: `sample_corpus_v2()`
   - Balances positive:negative ratio (e.g., 1:20) for training.
   - Output: corpus_{weak_num}_reduced_ratio_{ratio}.jsonl
4. **Prepare Formats**: Convert to JSONL for corpus and TSV for relevance labels.

### Data Loading

- Uses `GenericDataLoader` from BEIR for standard loading.
- Custom `WeakDataLoader` for combining original and weak data.

### Example Data Flow

- Raw data: corpus.jsonl (all documents), queries.jsonl, qrels.tsv
- Processed: Filtered corpus, sampled subsets, weak query files.

---

## Chapter 4: Soft Prompt Fine-Tuning

Soft prompt tuning is the core innovation of SPTAR, allowing efficient adaptation of LLMs for query generation.

### Overview

- **Goal**: Learn task-specific soft prompts that enable the LLM to generate relevant queries for given documents.
- **Efficiency**: Only ~0.003% of parameters are trainable (soft prompt embeddings).
- **Base Model**: Causal LLMs like LLaMA-7B, Vicuna-7B, GPT-2.

### Detailed Process (`xuyang/prompt_train_v1.py`)

1. **Model Loading**:
   - Load pretrained LLM and tokenizer.
   - Configure PEFT with `PromptTuningConfig`.
   - Set virtual tokens (e.g., 50) and initialization text (e.g., "please generate query for this document").

2. **Dataset Preparation** (`xuyang/dataset.py`):
   - Load labeled data (e.g., 50 query-document pairs).
   - Format inputs as: [Few-shot examples] + "Document: [text] \n Relevant Query: "
   - Use `MSMARCODataset` class for preprocessing.
   - Tokenize and pad sequences.

3. **Training**:
   - Optimizer: AdamW with linear warmup.
   - Loss: Cross-entropy on generated query tokens.
   - Early stopping based on validation perplexity.
   - Save best model (soft prompt weights).

4. **Hyperparameters**:
   - Batch size: 1 (for generation)
   - Epochs: 100
   - Learning rate: 3e-2
   - Virtual tokens: 50

### Key Components

- **PEFT Integration**: Uses Hugging Face's PEFT for soft prompt tuning.
- **Few-Shot Prompting**: Prepends 1-2 examples to guide generation.
- **Evaluation**: Monitors train/val perplexity and saves checkpoints.

---

## Chapter 5: Weak Query Generation (Augmentation)

Once soft prompts are tuned, they are used to generate weak queries for unlabeled documents.

### Overview

- **Input**: Unlabeled documents from filtered corpus.
- **Output**: Weak query-document pairs.
- **Quality**: Leverages tuned prompts for relevant query generation.

### Detailed Process (`xuyang/weak_inference.py`)

1. **Model Loading**:
   - Load fine-tuned soft prompt model using PEFT.
   - Prepare tokenizer.

2. **Generation Loop**:
   - For each document:
     - Construct prompt: [Few-shot examples] + "Document: [text] \n Relevant Query: "
     - Generate query using `model.generate()` with temperature=0.7.
     - Filter output (remove incomplete sentences).
     - Retry up to 3 times if generation fails.

3. **Output Formats**:
   - JSONL: Weak queries with IDs.
   - TSV: Query-ID, Corpus-ID, Score (1 for relevance).

4. **Hyperparameters**:
   - Max new tokens: 128
   - Prompt examples: 1-3 (configurable)
   - Fixed vs. Random prompts.

### Quality Control

- Simple filtering: Truncate at first punctuation if too long.
- Ensures generated queries are coherent and relevant.

---

## Chapter 6: Weak Data Filtering

Generated weak pairs may contain noise; filtering removes low-quality pairs.

### Overview

- **Method**: Use BM25 retrieval to verify if the generated query can retrieve its paired document.
- **Tool**: Pyserini (Anserini) for efficient BM25 indexing and search.

### Detailed Process (`zhiyuan/filter/bm25anserini_split.py`)

1. **Setup**:
   - Run Pyserini Docker container.
   - Index the corpus using the container's API.

2. **Retrieval Check**:
   - For each weak query, retrieve top-K documents using BM25.
   - Check if the paired document is in top-K.
   - If not, discard the pair.

3. **Batch Processing**:
   - Process in chunks (e.g., 5000 queries) for efficiency.
   - Use multi-threading if available.

4. **Output**:
   - Filtered JSONL and TSV files.
   - Logging: Number of filtered queries.

### Parameters

- **Top-K**: Typically 10-70 (configurable).
- **Reduction**: Reduces noise by ~20-50% depending on K.

---

## Chapter 7: Dense Retriever Training

The final step trains a dense retriever on the augmented dataset.

### Overview

- **Model**: Bi-encoder (e.g., BERT-based) from SentenceTransformers.
- **Data**: Original labeled + filtered weak pairs.
- **Goal**: Learn embeddings for queries and documents.

### Detailed Process (`zhiyuan/retriever/dpr/train/train_sbert.py`)

1. **Model Setup**:
   - Load SentenceTransformer model (e.g., bert-base-uncased).
   - Configure pooling and transformer layers.

2. **Data Loading**:
   - Use `WeakDataLoader` to combine original and weak data.
   - Prepare training samples: List of (query, positive_doc) pairs.

3. **Training**:
   - Loss: `MultipleNegativesRankingLoss`.
   - Batch size: 32
   - Optimizer: AdamW with warmup.
   - Evaluation: On dev set with IR metrics.

4. **Output**:
   - Trained model saved to disk.
   - Logs: Training loss, evaluation metrics.

### Variants

- Supports different retrievers: DPR, ColBERT, BM25CE.
- Scripts for each in `zhiyuan/retriever/`.

---

## Chapter 8: Loss Functions and Training Details

### Soft Prompt Tuning Loss

- **Type**: Cross-entropy loss on next-token prediction.
- **Formula**: \( \mathcal{L} = -\sum_{t=1}^{T} \log p(y_t | y_{<t}, x, \theta) \)
- **Where**: \( \theta \) are soft prompt embeddings, \( x \) is input, \( y \) is generated query.
- **Optimization**: AdamW, only \( \theta \) updated.

### Dense Retriever Loss

- **Type**: `MultipleNegativesRankingLoss` (contrastive loss).
- **Formula**: \( \mathcal{L} = -\log \frac{e^{\cos(q, d^+)}}{e^{\cos(q, d^+)} + \sum e^{\cos(q, d^-)}} \)
- **Where**: \( q \) is query embedding, \( d^+ \) is positive document, \( d^- \) are negatives (other positives in batch).
- **Benefit**: Efficient, no need for explicit negatives.

### Training Details

- **Soft Prompt**: Low LR (3e-2), small batch (1), long epochs (100).
- **DR**: Higher LR, larger batch (32), shorter epochs (20).

---

## Chapter 9: Key Files and Their Roles

| File/Folder                          | Role/Description                                                                                   |
|------------------------------------|--------------------------------------------------------------------------------------------------|
| `xuyang/prompt_train_v1.py`        | Main script for soft prompt tuning. Handles model loading, training loop, and saving.          |
| `xuyang/dataset.py`                 | Defines dataset classes for preprocessing inputs with few-shot prompts.                         |
| `xuyang/weak_inference.py`          | Generates weak queries using fine-tuned model. Includes filtering and output formatting.       |
| `zhiyuan/data_process.py`           | Core data processing: filtering, sampling, format conversion.                                   |
| `zhiyuan/filter/bm25anserini_split.py` | Filters weak data using BM25 retrieval via Pyserini.                                           |
| `zhiyuan/retriever/dpr/train/train_sbert.py` | Trains DPR model on augmented data using SentenceTransformers.                                 |
| `zhiyuan/dpr_eval.py`               | Orchestrates training and evaluation runs for DPR.                                              |
| `README.md`                        | Comprehensive guide with setup, commands, and results.                                          |
| `environment.yml`                  | Conda environments for different components.                                                     |
| `multi_run.sh`                     | Bash script for batch experiments.                                                               |

---

## Chapter 10: Running the Project: Step-by-Step Guide

1. **Setup Environments**:
   - Install dependencies: `conda env create -f environment.yml`
   - For ColBERT: Use col37bert environment.

2. **Data Preparation**:
   - Run `python zhiyuan/download.py` to download datasets.
   - Run `python zhiyuan/data_process.py` to process data.

3. **Soft Prompt Tuning**:
   - Command: `python xuyang/prompt_train_v1.py --dataset_name ms_50 --llm_name llama-7b --num_virtual_tokens 50`

4. **Weak Query Generation**:
   - Command: `python xuyang/weak_inference.py --peft_model_id [model_path] --data_path [corpus] --prompt_num 2`

5. **Weak Data Filtering**:
   - Start Pyserini: `docker run -p 8000:8000 beir/pyserini-fastapi`
   - Run: `python zhiyuan/filter/bm25anserini_split.py --dataset_name msmarco --topk 30`

6. **Dense Retriever Training**:
   - Run: `python zhiyuan/retriever/dpr/train/train_sbert.py --dataset_name msmarco --exp_name [exp] --weak_num 100k`

7. **Evaluation**:
   - Use `zhiyuan/dpr_eval.py` for batch evaluation.

---

## Chapter 11: Advanced Topics and Extensions

- **Joint Training**: Integrate query generation feedback into DR training for better alignment.
- **Hard Negatives**: Add mined hard negatives to improve contrastive loss.
- **Multi-Task Learning**: Combine with other IR tasks.
- **Scalability**: Use larger LLMs or distributed training.
- **Evaluation Metrics**: MRR@10, NDCG@10, Recall@100.

---

## Chapter 12: Conclusion

The SPTAR project provides a robust framework for augmenting dense retrieval with LLM-generated weak data. By understanding the detailed flow—from data preparation to final training—you can effectively run, modify, and extend the system. This guide serves as a comprehensive reference for navigating the repository and leveraging its capabilities for advanced IR research.

If you need code snippets, modifications, or further details on any section, refer back to the specific files or ask for assistance.
