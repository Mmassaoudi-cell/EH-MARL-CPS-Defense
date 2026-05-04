# EH-MARL: Hierarchical Multi-Agent RL for Adaptive CPS Defense

> **Paper:** "Multi-Agent Hierarchical Reinforcement Learning for Adaptive Cyber-Physical System Defense"  
> **Authors:** Mohamed Massaoudi, Maymouna Ez Eddin, Katherine R. Davis  
> **Affiliation:** Texas A&M University · Tarleton State University  
> **Support:** U.S. Department of Energy, Award DE-CR0000018

---

## Overview

This repository provides the full reproducible implementation of **EH-MARL** (Enhanced Hierarchical Multi-Agent Reinforcement Learning), a two-level defense framework for Cyber-Physical Systems (CPS).

| Level | Component | Role |
|---|---|---|
| 1 | Attention-based Coordinator | Selects which specialist to activate and allocates defense budget |
| 2 | 5× Dueling Double DQN Agents | Each specializes in one attack category (DoS, Probe, R2L, U2R, Normal) |

Key innovations over vanilla MARL:
- **Dueling Double DQN** with Prioritized Experience Replay (PER)
- **Multi-head self-attention** coordinator for context-aware agent routing
- **Entropy-regularized REINFORCE** to prevent premature policy collapse
- **SMOTE balancing** enabling 100% recall on rare R2L and U2R attacks

---

## Results

| Method | Accuracy | F1-Score | TTD (min) |
|---|---|---|---|
| Rule-based | 42.99% | 0.3590 | 40.6 |
| Single-Agent DQN | 99.06% | 0.9917 | 25.3 |
| Uncoordinated MARL | 100.0%* | 1.0000 | 22.1 |
| Hierarchical MARL | 98.08% | 0.9867 | 18.0 |
| **EH-MARL (Ours)** | **99.26%** | **0.9931** | **15.6** |

\* Perfect accuracy on balanced synthetic set only; see paper for transparency note.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments and generate figures
python main.py
```

Figures are saved to `figures/` (PDF + PNG). NSL-KDD is loaded automatically if present; otherwise realistic synthetic data is generated.

**Optional flags:**
```bash
python main.py --episodes_agent 150 --episodes_coord 200 --out figures/
```

---

## Repository Structure

```
.
├── main.py            # Full self-contained implementation (single entry point)
├── requirements.txt   # Pinned dependencies
└── figures/           # Generated after running main.py (8 publication figures)
```

---

## Requirements

- Python 3.10+
- PyTorch >= 2.1
- scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{massaoudi2025ehmarl,
  title   = {Multi-Agent Hierarchical Reinforcement Learning for
             Adaptive Cyber-Physical System Defense},
  author  = {Massaoudi, Mohamed and Ez~Eddin, Maymouna and
             Haque, Khandaker Akramul and Davis, Katherine R.},
  journal = {IEEE (submitted)},
  year    = {2025}
}
```

---

## License

This project is for academic use. Contact `mohamed.massaoudi@tamu.edu` for reuse inquiries.
