# Quantization-Robust LLM Unlearning via Low-Rank Adaptation

**João Vitor Boer Abitante, Joana Meneguzzo Pasquali, Luan Fonseca Garcia, Ewerton de Oliveira, Thomas da Silva Paula, Rodrigo C. Barros, Lucas S. Kupssinskü**

Accepted at **IJCNN 2026** · [arXiv:2602.13151](https://arxiv.org/abs/2602.13151)

This repository implements a framework for **Quantization-Robust Machine Unlearning** in large language models, with a focus on **Llama-2-7B** and **LoRA-based unlearning**. It accompanies the paper "Quantization-Robust LLM Unlearning via Low-Rank Adaptation" and targets a known failure mode: unlearning updates that vanish under **4-bit quantization**.

---

## Project Overview

**Problem:** Standard unlearning updates can exhibit *catastrophic failure* after 4-bit quantization, where small weight changes are rounded away and the model largely reverts to its pre-unlearning behavior.

**Hypothesis:** By restricting optimization to a **low-rank subspace (LoRA)**, the unlearning signal becomes concentrated in fewer directions, producing updates that are sufficiently large to **cross quantization boundaries**. This makes the unlearning effects persist after **RTN (round-to-nearest) quantization**.

---

## Key Technical Mechanisms

- **Optimization Dynamics:** We use **significantly higher learning rates** (eta ~= 1e-4) than full-parameter fine-tuning. LoRA's implicit regularization allows this without destabilizing training.
- **Magnitude Control:** We explicitly scale updates via **alpha** and target **all linear layers** (Attention projections and MLP up/down/gate projections) to ensure updates survive 4-bit discretization noise.

---

## Methods Supported

Unlearning objectives and utility preservation strategies implemented in this codebase include:

- **Gradient Ascent (GA)**
- **Negative Preference Optimization (NPO)**
- **Gradient Descent on Retain Set (GDR)**
- **KL Minimization to Retain Model (KLR)**

We primarily report combinations: **GA+GDR**, **GA+KLR**, **NPO+GDR**, and **NPO+KLR**.

---

## Experimental Setup

- **Benchmark:** MUSE (News and Books datasets)
- **Model:** Llama-2-7B
- **Metrics:**
  - **VerMem** (Verbatim Memorization, lower is better)
  - **KnowMem** (Knowledge Memorization, lower is better)
  - **PrivLeak** (Privacy Leakage, closer to 0 is better)
  - **Utility** (higher is better)

---

## Installation and Usage

### Requirements (placeholders)

Install dependencies with your preferred environment manager.

Refer to `requirements.txt` for a minimal set of packages used in this repository.

### Example Usage

The following script runs LoRA-based unlearning for MUSE and evaluates at multiple precisions:

```bash
bash scripts/paper_muse_unlearn_lora.sh
```

**Important:** LoRA adapters **must be merged into the base model parameters before quantization** (RTN). This ensures the unlearning updates persist after 4-bit discretization.

---

## Results Summary

The following results demonstrate that **LoRA improves quantization robustness**, keeping unlearning metrics stable even after aggressive 4-bit quantization.

### Experimental Results: Quantization Robustness (4-bit)

| Method | Prec. / Adapter | BOOKS: VerMem (down) | BOOKS: KnowMem (down) | BOOKS: PrivLeak (to 0) | BOOKS: Utility (up) | NEWS: VerMem (down) | NEWS: KnowMem (down) | NEWS: PrivLeak (to 0) | NEWS: Utility (up) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GA+GDR** | Full | 0.00 | 36.30 | -24.01 | 68.74 | 52.15 | 56.98 | -99.79 | 49.57 |
| | Full + **LoRA** | 0.00 | 37.68 | -3.79 | 61.90 | 46.49 | 52.13 | -99.79 | 47.78 |
| | 4-bit | 0.00 | 28.77 | -23.65 | 53.79 | 42.07 | 48.19 | -99.79 | 40.06 |
| | 4-bit + **LoRA** | 0.00 | 26.43 | -3.77 | 53.16 | 40.22 | 48.15 | -99.79 | 44.82 |
| **GA+KLR** | Full | 0.00 | 34.62 | -24.66 | 62.14 | 49.01 | 63.12 | -99.51 | 52.14 |
| | Full + **LoRA** | 0.00 | 0.00 | -3.67 | 62.19 | 52.33 | 60.11 | -99.74 | 52.29 |
| | 4-bit | 0.00 | 23.64 | -25.68 | 44.13 | 43.38 | 53.24 | -99.51 | 44.18 |
| | 4-bit + **LoRA** | 0.14 | 0.00 | -5.86 | 50.30 | 41.72 | 53.68 | -99.74 | 47.77 |
| **NPO+GDR** | Full | 54.61 | 33.39 | -56.37 | 60.09 | 26.89 | 52.11 | -86.04 | 48.90 |
| | Full + **LoRA** | 22.67 | 36.63 | -60.07 | 59.65 | 46.39 | 59.51 | -99.74 | 48.61 |
| | 4-bit | 41.18 | 25.64 | -58.45 | 50.17 | 23.91 | 47.63 | -87.53 | 44.01 |
| | 4-bit + **LoRA** | 20.30 | 36.64 | -58.91 | 58.10 | 37.78 | 49.09 | -99.72 | 46.40 |
| **NPO+KLR** | Full | 51.39 | 31.16 | -55.82 | 60.25 | 24.03 | 45.81 | -86.85 | 48.13 |
| | Full + **LoRA** | 16.76 | 26.48 | -61.32 | 41.82 | 35.67 | 48.30 | -94.73 | 40.89 |
| | 4-bit | 38.65 | 26.00 | -57.87 | 48.50 | 22.09 | 46.80 | -87.63 | 44.76 |
| | 4-bit + **LoRA** | 17.03 | 24.33 | -56.88 | 42.02 | 28.24 | 48.40 | -95.42 | 39.96 |

---

## Citing this work

If you use this repository in your research, please cite our paper:

```bibtex
@misc{abitante2026quantizationrobustllmunlearninglowrank,
      title={Quantization-Robust LLM Unlearning via Low-Rank Adaptation}, 
      author={João Vitor Boer Abitante and Joana Meneguzzo Pasquali and Luan Fonseca Garcia and Ewerton de Oliveira and Thomas da Silva Paula and Rodrigo C. Barros and Lucas S. Kupssinskü},
      year={2026},
      eprint={2602.13151},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.13151}, 
}
```

---

## Acknowledgements

This repository is built on top of [OpenUnlearning](https://github.com/open-unlearning/open-unlearning), an open-source framework for LLM unlearning developed by Vineeth Dorna ([@Dornavineeth](https://github.com/Dornavineeth)) and Anmol Mekala ([@molereddy](https://github.com/molereddy)). We thank them and all contributors to that project for providing a well-structured and extensible codebase that made this work possible.

---

## License

See `LICENSE` for the full license text.
