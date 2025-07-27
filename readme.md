# A Modular Framework for Comparing PEFT Methods in Retrieval-Augmented Generation

**A systematic framework for fine-tuning and evaluating multiple Large Language Models (LLMs) using various Parameter-Efficient Fine-Tuning (PEFT) techniques on a Retrieval-Augmented Generation (RAG) task.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers%20%7C%20PEFT-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 1. Project Overview

This project introduces a powerful, automated framework designed to systematically benchmark the trade-offs between traditional fine-tuning and modern Parameter-Efficient Fine-Tuning (PEFT) methods for Retrieval-Augmented Generation (RAG). While full fine-tuning is computationally prohibitive, PEFT techniques like LoRA, IA³, and AdaLoRA promise to democratize LLM customization by training only a tiny fraction of the model's parameters.

This framework tests that promise by answering a critical question: **How much performance do we sacrifice for a massive gain in efficiency?**

To do this, the project automates the entire experimental pipeline:
- **Configuration:** A single `config.yaml` file defines a matrix of experiments, comparing multiple base LLMs (`gpt2`, `distilgpt2`, `GPT-Neo-125M`) against various tuning methods.
- **Execution:** The framework automatically runs each experiment, fine-tuning the models on the SQuAD dataset within a RAG context.
- **Analysis:** It generates a suite of publication-quality visualizations that directly compare each method on performance, training time, and parameter cost.

---

## 2. Results & Analysis Dashboard

The automated analysis generates four key visualizations, providing a comprehensive overview of the experimental results.

| Performance vs. Efficiency | Performance Metrics Comparison |
| :---: | :---: |
| ![Performance vs Efficiency](plot_1_performance_vs_efficiency.png) | ![Performance Comparison](plot_2_performance_comparison.png) |
| **Cost Analysis (Time & Parameters)** | **Model-wise Deep Dive** |
| ![Computation Cost](plot_3_cost_comparison.png) | ![Model-wise Comparison](plot_4_model_deep_dive.png) |

### Key Findings from the Analysis

Based on the experimental data, we can draw several specific conclusions:

1.  **PEFT is Drastically More Efficient:** PEFT methods (LoRA, IA³, AdaLoRA) achieved comparable or even superior performance to Full Fine-Tuning while training **less than 1%** of the model's total parameters. This represents a monumental reduction in computational cost.

2.  **Full Fine-Tuning Underperforms in Low-Data Regimes:** With only a single training epoch, the Full Fine-Tuning (Full FT) baselines performed poorly, achieving near-zero scores. This strongly suggests that PEFT methods are not only more efficient but also more effective at adapting to a new task quickly with limited data.

3.  **`GPT-Neo-125M` with `LoRA` is the Champion:** In this specific test, the `GPT-Neo-125M_LoRA` configuration emerged as the top performer, achieving the highest Semantic Accuracy score (`0.313`). This indicates a strong synergy between the GPT-Neo architecture and the LoRA adapter for this RAG task.

4.  **Semantic Accuracy Tells a Different Story:** While traditional metrics like ROUGE and BLEU showed minimal variation, the custom **Semantic Accuracy** metric revealed significant performance differences between methods, highlighting its value in evaluating the true meaning of generated text.

---

## 3. How It Works

The framework's architecture is designed for modularity and automation.


[config.yaml] -> [main.py] --(Loads Data)--> [RAG Pipeline (FAISS)]
|
--(For each experiment)--> [Model Loader (Quantization + PEFT)] | --> [Trainer] --(Fine-Tunes)--> [Evaluator]
|
--> [results.csv] | --> [analyze_results.py] -> [Plots]


1.  **Configuration (`config.yaml`):** Defines the entire suite of experiments, including base models, adapter types, and hyperparameters.
2.  **RAG Setup:** At the start, the context from the SQuAD dataset is vectorized using Sentence-Transformers and indexed into a FAISS database for fast similarity search.
3.  **Model Loading:** For each experiment, the appropriate base LLM is loaded. If a PEFT method is used, the model is quantized to 4-bit precision to save memory. For full fine-tuning, it is loaded in half-precision (16-bit). The specified adapter is then attached.
4.  **Training Loop:** The `Trainer` iterates through the dataset. For each question, the RAG pipeline retrieves the most relevant context, which is then used to build the prompt for the LLM. Only the adapter weights (or all weights, for full FT) are updated.
5.  **Evaluation & Logging:** After training, the model's performance is measured against a test set using ROUGE, BLEU, and the custom Semantic Accuracy metric. All results, including training time and parameter counts, are logged to `results.csv`.
6.  **Automated Analysis:** The `analyze_results.py` script processes `results.csv` to generate the final suite of plots, providing immediate insight into the experimental outcomes.

---

## 4. Setup and Usage

### 4.1. Prerequisites
- Python 3.10+
- A CUDA-enabled GPU is highly recommended, especially for the full fine-tuning experiments.

### 4.2. Installation
1.  Clone the repository:
    ```bash
    git clone ![https://github.com/aks-akanksha/adapter-based-fine-tuning-for-RAG.git](https://github.com/aks-akanksha/adapter-based-fine-tuning-for-RAG.git)
    cd adapter-based-fine-tuning-for-RAG
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 4.3. Running the Framework

The entire workflow is automated into two simple steps.

1.  **Execute the Experiment Suite:**
    Run the main script. It will automatically loop through all experiments defined in `config.yaml`, run them, and save the results.
    ```bash
    python main.py
    ```
    The script is designed to be resumable. If it's interrupted, you can run it again, and it will automatically skip any experiments that are already logged in `results.csv`.

2.  **Generate the Analysis Dashboard:**
    Once all experiments are complete, run the analysis script:
    ```bash
    python analyze_results.py
    ```
    This will print a summary table to your console and save four high-quality plots (`plot_1_...`, `plot_2_...`, etc.) to your project directory.

---

## 5. Future Work

This framework provides a strong foundation for further research. Potential next steps include:
-   **Integrating Larger Models:** Testing more powerful, modern LLMs like Mistral-7B or Llama-3-8B.
-   **Exploring Advanced RAG:** Implementing more sophisticated retrieval techniques, such as multi-query retrievers or re-ranking models.
-   **Expanding PEFT Methods:** Adding support for other promising PEFT techniques like VeRA or (IA)³.
-   **Broader Task Evaluation:** Adapting the framework to evaluate performance on different NLP tasks, such as summarization or translation.
