# Medical Question Answering — PubMedQA  
### EMATM0067 Introduction to AI and Text Analytics · Spring 2026 · University of Bristol

---

## Overview

This repository contains the full implementation for **Task 1: Medical Question Answering**. The objective is to predict whether the answer to a biomedical research question is:

- `yes`
- `no`
- `maybe`

given the question and its supporting PubMed abstract.

We explore two key design axes:

- **Axis I — Text Representation**  
  TF-IDF → BM25 → Word2Vec → PubMedBERT (frozen) → PubMedBERT (fine-tuned)

- **Axis II — Answer Strategy**  
  Discriminative classification vs. GPT few-shot generation

---

## Dataset

We use the **PubMedQA** dataset, consisting of:

| Split | Size | Description |
|-------|------|-------------|
| PQA-L | 1,000 | Gold-labelled (training & evaluation) |
| PQA-U | 61,333 | Unlabelled (self-training) |
| PQA-A | 211,269 | Artificial (weak pre-training) |

**Class distribution (PQA-L):**
- yes: 55.2%  
- no: 33.8%  
- maybe: 11.0%

---


