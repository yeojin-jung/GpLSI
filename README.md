# 📘 GpLSI: Topic Modeling with Document Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **GpLSI** from the paper:

> **Graph Topic Modeling for Documents with Spatial or Covariate Dependencies**  
> Y.J. Jung and C. Donnat
> [[arXiv:2412.14477]](https://arxiv.org/pdf/2412.14477)

This repository provides:

- A cleaned and installable **Python package** `gplsi`
- Code for **synthetic experiments**
- Code for **three real-world datasets** (cellular microenvironments (CODEX), What's cooking?)

---

## 🌟 Key Features

- **Graph-regularized topic model (GpLSI)** based on **iterative Graph-Aligned SVD**, designed for nonnegative matrix factorization of frequency matrices with an underlying document graph (spatial or covariate-defined).
- **End-to-end spatial omics workflow** with CODEX datasets that demonstrates 
  - how to choose the number of topics (K) using consistency across batches
  - recover and interpret spatially coherent topics
  -	perform downstream prediction tasks (e.g., survival analysis) using the learned topic proportions
- Comparison with existing methods (pLSI, LDA, spatial LDA variants, TopicScore)
- Integrates with:
  - **Anchor-based topic recovery** (Successive Projection Overlapping Clustering (SPOC) from Klopp et al., 2021)
- Works for:
  - Spatial transcriptomics  
  - Cellular microenvironments
  - Texts with metadata
---

## 📦 Installation

We recommend using the provided conda environment for a clean and reproducible setup.

#### 1. Create the environment

From the repository root:
```bash
conda env create -f environment.yml
conda activate gplsi-env
```

#### 2. Install GpLSI
Clone the repository and install in editable mode:
```bash
git clone https://github.com/yeojin-jung/GpLSI.git
cd GpLSI
pip install -e .
```
This will make the package importable as:
```bash
from gplsi import GpLSI
```

#### 3. Install pycvxcluster 
`pycvxcluster` is required for the graph-regularized update step used in GpLSI. Install it manually:
```bash
git clone https://github.com/signal-lab-uchicago/pycvxcluster-0.1.0.git
cd pycvxcluster-0.1.0
pip install -e .
```

#### 4. Final sanity check
Back in the `GpLSI` repo:
```bash
python -c "from gplsi import GpLSI; import pycvxcluster.pycvxcluster; print('OK')"
```

If this prints 'OK', you are fully set up!

## 📂 Repository Structure
```bash
GpLSI/
│
├── src/gplsi/
│   ├── gplsi.py                # Main GpLSI class
│   ├── graphSVD.py             # Graph-aligned SVD backend
│   ├── utils.py                # Internal utilities
│   ├── generate_topic_model.py # Synthetic data generator
│   ├── realdata_spleen.py      # Spleen real-data pipeline
│   ├── realdata_crc.py         # CRC real-data pipeline
│   ├── realdata_cook.py        # What’s Cooking pipeline
│
├── codes/                      # Run synthetic/real data experiments
├── data/                       # Raw datasets
│
├── tutorial.ipynb              # 📘 Complete usage walkthrough
└── README.md
```
## 🧠 Basic Usage
Fit GpLSI to any frequency dataset
```bash
import numpy as np
from gplsi import GpLSI

model = GpLSI(
    lamb_start=1e-4,
    step_size=1.25,
    grid_len=29,
    eps=1e-5,
)

model.fit(
    X,          # (n × p) row-normalized frequency matrix
    N,          # mean row sum of original counts
    K,          # number of topics
    edge_df,    # DataFrame with columns: (src, tgt, weight)
    weights,    # sparse adjacency matrix
)

W = model.W_hat   # topic proportions (n × K)
A = model.A_hat   # topic loadings    (K × p)
```

## 🧪 Running Experiments
All experiment scripts live under codes/ and call the public APIs in src/gplsi.

### 1️⃣ Synthetic Simulation Experiments
After installing the package:
```bash
cd codes/
python run_sim.py --task_id 1
```
This will run an experiment with configurations stored in the first row of `codes/config.txt`. 
Or manually specify configurations:
```bash
python run_sim.py \
    --nsim 50 --N 100 --n 1000 --p 30 --K 3
```
### 2️⃣ Real Data: Mouse Spleen (CODEX)
We use data from the study “Deep Profiling of Mouse Splenic Architecture with CODEX Multiplexed Imaging” by Goltsev et al., published in Cell (2018, Volume 174, Issue 4, pp. 968–981.e15).

```bash
python run_spleen.py \
    --K 6 \
    --lamb_start 1e-4 \
    --step_size 1.25 \
    --grid_len 29 \
    --eps 1e-5 \
    --tumor 0   # (BALBc-1)
```
### 3️⃣ Real Data: Stanford CRC (CODEX)
```bash
python run_crc.py \
    --K 6 \
    --lamb_start 1e-4 \
    --step_size 1.25 \
    --grid_len 29 \
    --eps 1e-5
```

For this example, we provide a method to choose the **optimal number of topics** by measuring consistency of estimated topics across batches of patients. 
```bash
# run for K = 2,3,...
mpiexec -n 5 python run_crc_choose_ntopics.py \
    --K 6 \ 
    --lamb_start 1e-4 \
    --step_size 1.25 \
    --grid_len 29 \
    --eps 1e-5
```

We also provide the code for postprocessing the result to conduct survival analysis (patient outcome prediction).
```bash
python postprocess_crc.py \
```

## 📚 External Methods Included in This Repository

This repository includes implementations of two external baseline methods used for model comparison in our experiments. These components are included for reproducibility and fair benchmarking against GpLSI.

 1. TopicSCORE ("A new SVD approach to optimal topic estimation" by Tracy Ke and Minzhe Wang (2017))

 `utils/topicscore.r`
 
 Original repository: https://github.com/ZhengTracyKe/TopicSCORE

 This R script is provided as-is from the TopicSCORE authors to allow comparison with their spectral method for topic modeling.
 
 2. Spatial LDA ("Modeling Multiplexed Images with Spatial-LDA Reveals Novel Tissue Microenvironments" by Zhenghao Chen et al. (2020))

  `utils/spatial_lda/`

 Original repository: https://github.com/calico/spatial_lda

This directory contains the authors’ implementation of Spatial-LDA.
To integrate it into our benchmarking pipeline, we made minimal modifications while preserving the original algorithm and logic. All credit for the underlying method belongs to the original authors.


## 📜 Citation

If you use this code, please cite:
```bash
@article{jung2024gplsi,
  title={Graph Topic Modeling for Documents with Spatial or Covariate Dependencies},
  author={Jung, Yeo Jin and Donnat, Claire},
  journal={arXiv preprint arXiv:2412.14477},
  year={2025}
}
}
```
## 🔗 Links
- 📦 PyPI: 
- 💻 GitHub: https://github.com/yeojin-jung/GpLSI
- 📑 Paper: https://arxiv.org/pdf/2412.14477
