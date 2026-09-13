# Logit Laundering: Evading Data-Use Auditing in Large Language Models

本仓库为 [Logit Laundering: Evading Data-Use Auditing in Large Language Models](‼️PAPER_URL)（ACM CCS 2026）的官方实现，作者：Ruihan Hu, Wei Luo, Yu-Ming Shang, Jiakai Wang, Chuxuan Zhang, Mei Li, Xi Zhang, Meikang Qiu。

[项目主页](docs/index.html) | [Paper](‼️PAPER_URL) | [English](README.md)

## Overview

我们研究 LLM 中的 **data-use auditing** 问题：在只能黑盒访问模型的前提下，审计方能否判定提供方在未经授权、带水印的数据上训练？

我们提出 **Logit Laundering**，一种 **model-level** 规避框架。与训练前破坏语料的数据级规避不同，轻量 **Statistical Bias Extractor (SBE)** 在推理阶段修正 next-token logits，在 Jensen–Shannon 散度预算下削弱 STAMP/Radioactive 可见的水印统计信号，且无需重训 base model。图表与完整结果见[项目主页](docs/index.html)。

每步解码：$z_t^{\mathrm{Bob}} = z_t^{\mathrm{base}} - \lambda_t \, z_t^{\mathrm{SBE}}$，其中 $\lambda_t$ 由 base model 的 top-1/top-2 logit margin 自适应确定。

:star: 如果本工作对您有帮助，欢迎引用 :star:

```bibtex
@inproceedings{hu2026logitlaundering,
  title={Logit Laundering: Evading Data-Use Auditing in Large Language Models},
  author={Hu, Ruihan and Luo, Wei and Shang, Yu-Ming and Wang, Jiakai and Zhang, Chuxuan and Li, Mei and Zhang, Xi and Qiu, Meikang},
  booktitle={Proceedings of the ACM Conference on Computer and Communications Security (CCS)},
  year={2026}
}
```

## 📘 Data

流水线读取至少含 **`text`** 字段的 **JSON/JSONL**。在 `configs/experiment.env` 中配置路径：

```bash
cp configs/experiment.env.example configs/experiment.env
```

* **`SOURCE_DATA`**：你的授权训练文本（水印注入的输入）。
* **`WATERMARKED_DATA` / `AUDIT_DATA`**：由 `watermark` 阶段生成。
* **`MCQ_DATA` / `UTILITY_DATA`**：本地准备的评估集。

论文实验涉及的版权**书籍语料不予公开**。方法与完整流水线均已发布——请用合法授权数据复现。

## 🚀 Run Logit Laundering

**环境要求：** NVIDIA GPU、CUDA、Python 3.10+。

本仓库复现论文中的 **五个基座模型**：

- **LLaMA：** `meta-llama/Llama-3.2-1B-Instruct`、`Llama-3.2-3B-Instruct`
- **Qwen：** `Qwen/Qwen2.5-1.5B-Instruct`
- **Gemma：** `google/gemma-2-2b-it`
- **StableLM：** `stabilityai/stablelm-2-1_6b-chat`

**审计器：** STAMP、Radioactive、Top-1。**效用评估：** MCQ、采样统计、lm-eval。

#### Installation

```bash
git clone https://github.com/STAIR-BUPT/Logit-Laundering.git REPO && cd REPO

python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r env/requirements.txt
pip install -e . && pip install -e base_model_training/llamafactory
pip install -e utility_evaluation/lm_eval_harness
```

🔐 **Important：** Base 训练用仓库内 `base_model_training/llamafactory`。**SBE 训练**需单独 patched clone——记路径为 `LF2`，并在 `experiment.env` 中设 `LLAMA_FACTORY2_ROOT`：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git LF2
git -C LF2 checkout "$(tr -d '[:space:]' < REPO/assistant_model/llamafactory2_training/patches/LLaMA-Factory2_BASE_HEAD.txt)"
bash REPO/assistant_model/llamafactory2_training/scripts/apply_llamafactory2_patch.sh LF2
pip install -e LF2
```

在 LLaMA-Factory 的 `dataset_info.json` 中注册训练 JSONL。Assistant 训练时，目标子集放 JSON **最前**，并设 **`EVASION_SIZE`** 为其长度。

#### Run the pipeline

每个 stage 一条命令，**按顺序**执行：

```bash
logit-laundering run watermark
logit-laundering run train-base
logit-laundering run build-assistant
logit-laundering run train-assistant
logit-laundering run calibrate
logit-laundering run top1
```

🔍 **Stages explained:**

* **watermark** — 注入水印，构造带水印训练语料与审计语料。
* **attack** *（可选）* — 训练前的数据级 evasion baseline。
* **train-base** — 公共语料 + 带水印语料持续预训练。
* **build-assistant** — 从训好的 base model 初始化轻量 SBE。
* **train-assistant** — SBE 双目标微调（拟合水印偏置、压平无关偏置）。
* **calibrate** — JS 预算下估计 $\lambda$；将 **`alpha_max`** 写回 `experiment.env`。
* **top1** / **stamp** — Top-1 与 STAMP 审计（规避 vs. 检测 p-value）。
* **mcq** / **sampling** / **lm-eval** — 知识保留、生成统计与通用能力。

📌 **Note：** `logit-laundering list-stages` 列出全部 stage；公开结果前运行 `logit-laundering validate`。上文示例之后还需运行 `stamp`、`mcq`、`sampling`、`lm-eval`。可选：`attack`。

---

[MIT License](LICENSE)。仅供 data-use auditing 鲁棒性研究。
