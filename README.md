# Logit Laundering: Evading Data-Use Auditing in Large Language Models

This repository provides the official implementation of Logit Laundering: Evading Data-Use Auditing in Large Language Models (ACM CCS 2026) by Ruihan Hu, Wei Luo, Yu-Ming Shang, Jiakai Wang, Chuxuan Zhang, Mei Li, Xi Zhang, and Meikang Qiu.

[Project Page](docs/index.html) | [中文](README.zh-CN.md)

## Overview

We study the **data-use auditing** problem in LLMs: given black-box access to a model, can an auditor determine whether the provider trained on watermarked, unauthorized data?

We propose **Logit Laundering**, a **model-level** evasion framework. Instead of corrupting training corpora (data-level evasion), a lightweight **Statistical Bias Extractor (SBE)** reshapes next-token logits at inference—suppressing STAMP/Radioactive-visible watermark statistics under a Jensen–Shannon divergence budget, without retraining the base model. See our [project page](docs/index.html) for figures and full results.

At each decoding step: $z_t^{\mathrm{Bob}} = z_t^{\mathrm{base}} - \lambda_t \, z_t^{\mathrm{SBE}}$, where $\lambda_t$ adapts to the base model's top-1/top-2 logit margin.

## 📘 Data

The pipeline reads **JSON/JSONL** files with at least a `text` field. Point paths in `configs/experiment.env`:

```bash
cp configs/experiment.env.example configs/experiment.env
```

* **`SOURCE_DATA`**: your authorized training text (input to watermarking).
* **`WATERMARKED_DATA` / `AUDIT_DATA`**: produced by the `watermark` stage.
* **`MCQ_DATA` / `UTILITY_DATA`**: evaluation sets you prepare locally.

Copyright-protected **book corpora** from our paper experiments are **not released**. The full method and pipeline are public—use your own licensed data to reproduce the workflow.

## 🚀 Run Logit Laundering

**Requirements:** NVIDIA GPU, CUDA, Python 3.10+.

Our codebase reproduces the paper on **five foundation models**:

- **LLaMA:** `meta-llama/Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`
- **Qwen:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Gemma:** `google/gemma-2-2b-it`
- **StableLM:** `stabilityai/stablelm-2-1_6b-chat`

**Auditors:** STAMP, Radioactive, Top-1. **Utility:** MCQ accuracy, sampling statistics, lm-eval.

#### Installation

```bash
git clone https://github.com/STAIR-BUPT/Logit-Laundering.git REPO && cd REPO

python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r env/requirements.txt
pip install -e . && pip install -e base_model_training/llamafactory
pip install -e utility_evaluation/lm_eval_harness
```

🔐 **Important:** Base-model training uses the in-repo LLaMA-Factory under `base_model_training/llamafactory`. **SBE training** needs a separate patched clone—set its path as `LF2` and `LLAMA_FACTORY2_ROOT` in `experiment.env`:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git LF2
git -C LF2 checkout "$(tr -d '[:space:]' < REPO/assistant_model/llamafactory2_training/patches/LLaMA-Factory2_BASE_HEAD.txt)"
bash REPO/assistant_model/llamafactory2_training/scripts/apply_llamafactory2_patch.sh LF2
pip install -e LF2
```

Register training JSONL in LLaMA-Factory `dataset_info.json`. For assistant training, place the target subset **first** in the JSON and set **`EVASION_SIZE`** to its length.

#### Run the pipeline

Each stage is one command. Run them **in order**:

```bash
logit-laundering run watermark
logit-laundering run train-base
logit-laundering run build-assistant
logit-laundering run train-assistant
logit-laundering run calibrate
logit-laundering run top1
```

🔍 **Stages explained:**

* **watermark** — Inject watermarks; build watermarked training and audit corpora.
* **attack** *(optional)* — Data-level evasion baseline before training.
* **train-base** — Continual pretraining on mixed public + watermarked data.
* **build-assistant** — Initialize the lightweight SBE from the trained base model.
* **train-assistant** — Dual-objective SBE fine-tuning (fit watermark bias, flatten unrelated bias).
* **calibrate** — Estimate $\lambda$ under the JS budget; write **`alpha_max`** back to `experiment.env`.
* **top1** / **stamp** — Top-1 and STAMP auditing (evasion vs. detection p-values).
* **mcq** / **sampling** / **lm-eval** — Knowledge retention, generation statistics, and general utility.

📌 **Note:** Use `logit-laundering list-stages` to list all stages and `logit-laundering validate` before sharing results. After the commands above, also run `stamp`, `mcq`, `sampling`, and `lm-eval`. Optional: `attack`.

---

Released under the [MIT License](LICENSE). For research on robust data-use auditing only.
