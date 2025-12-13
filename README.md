
---

# Optimizing Text-Based Person Search (TBPS) with Circle Loss & mSigLIP


## Abstract

Text-Based Person Search (TBPS) aims to retrieve person images given natural language descriptions. While recent vision–language models such as SigLIP and its multilingual variant mSigLIP have shown strong cross-modal representation capabilities, their performance degrades when applied to low-resource languages such as Vietnamese. In this work, we investigate the limitations of the commonly used Normalized Image-Text Contrastive (N-ITC) loss under Vietnamese TBPS settings and propose integrating Circle Loss to enable adaptive hard-sample mining. We design and evaluate four integration strategies that combine Circle Loss with mSigLIP at both unimodal and cross-modal levels. Experiments on the VN3K dataset demonstrate consistent improvements over the baseline, particularly in Rank-1 accuracy and mAP, validating the effectiveness of Circle Loss for addressing both cross-modal and cross-lingual gaps.

---

## 1. Introduction

Text-Based Person Search (TBPS) is a challenging cross-modal retrieval task that requires aligning visual person representations with textual descriptions. Recent approaches based on large-scale vision–language pretraining have achieved promising results; however, most are optimized for high-resource languages. For Vietnamese TBPS, two major challenges arise: (1) the cross-modal gap between image and text embeddings, and (2) the cross-lingual gap caused by linguistic differences from the pretraining corpus.

The baseline TBPS-mSigLIP model employs the Normalized Image-Text Contrastive (N-ITC) loss. Although effective in general settings, N-ITC applies uniform penalties to all samples, limiting its ability to emphasize hard positives and hard negatives. This motivates the adoption of Circle Loss, which introduces adaptive re-weighting based on sample difficulty.

---

## 2. Methodology

### 2.1 Problem Formulation

Given an image–text pair $(I, T)$, TBPS aims to learn a joint embedding space where matched pairs have high similarity and mismatched pairs have low similarity. Let $s_p$ and $s_n$ denote similarity scores for positive and negative pairs, respectively.

### 2.2 Circle Loss

Circle Loss introduces adaptive weighting factors that dynamically adjust the contribution of each similarity score during optimization. The loss is defined as:

$$
\mathcal{L}_{circle} = \log\left[1 + \sum \exp(\gamma \alpha_n s_n) \sum \exp(-\gamma \alpha_p s_p)\right]
$$

where $\gamma$ is a scale factor and $\alpha_p, \alpha_n$ control the relative importance of positive and negative pairs. This formulation encourages $s_p \to 1$ and $s_n \to 0$, while focusing training on hard samples.

---

## 3. Integration Strategies

We explore four strategies for integrating Circle Loss into the TBPS-mSigLIP framework:

### Strategy 1: In-Modal Auxiliary Optimization

Circle Loss is applied independently to the image and text branches as an auxiliary regularizer. This strategy promotes identity-level clustering within each unimodal embedding space prior to cross-modal alignment.

### Strategy 2: Intrinsic N-ITC (NC-ITC)

The sigmoid-based logit computation in N-ITC is modified using Circle Loss re-weighting, allowing direct optimization of the decision boundary.

### Strategy 3: Pure Cross-Modal Circle Loss

N-ITC is fully replaced by Cross-Modal Circle Loss, yielding embeddings with high inter-class separability.

### Strategy 4: Hybrid N-ITC with Cross-Modal Auxiliary (Selected)

The final strategy combines N-ITC as the main objective with an auxiliary Cross-Modal Circle Loss branch dedicated to hard-sample mining. Multi-View Supervision (MVS) is applied to both branches. This hybrid formulation achieves the best empirical performance and is adopted for final evaluation.

---

## 4. Implementation Details

The project is implemented in PyTorch and structured as follows:

* **model/objectives.py**: Loss function implementations, including pairwise Circle Loss, intrinsic N-ITC, and cross-modal Circle Loss.
* **tbps.py**: Main model definition, forward pipeline, and loss strategy selection.

Additional techniques include LoRA-based fine-tuning, self-supervision, and multi-view supervision.

**Hardware:** Intel Core i5-7600K, NVIDIA RTX 3060 (12GB), 24GB RAM.

---

## 5. Experiments

### 5.1 Datasets

Experiments are conducted on the 3000VNPersonSearch dataset, with additional evaluation on CUHK-PEDES for generalization analysis.

### 5.2 Evaluation Metrics

We report Rank@1, Rank@5, Rank@10, mean Average Precision (mAP), and mean Inverse Negative Penalty (mINP).

### 5.3 Results

The hybrid strategy consistently outperforms the baseline TBPS-mSigLIP model:

```bash
+------+-------+-------+-------+-------+-------+
| Task |   R1  |   R5  |  R10  |  mAP  |  mINP |
+------+-------+-------+-------+-------+-------+
| t2i  | 50.53 | 77.78 | 86.43 | 55.94 | 49.37 |
| i2t  | 52.35 | 77.55 | 86.20 | 48.97 | 33.27 |
+------+-------+-------+-------+-------+-------+
```

* Rank@1: +0.83%
* Rank@5: +1.85%
* Rank@10: +1.68%
* mAP: +0.98%
* mINP: +0.71%

Circle Loss parameters are set to Margin = 0.25 and Gamma = 128, combined with LoRA fine-tuning.

---

## References

* Sun et al., *Circle Loss: A Unified Perspective of Pair Similarity Optimization*, CVPR 2020.

---

## Setup

### 0. Clone the repository

```bash
git clone https://github.com/pahmlam/Research_on_CircleLoss_for_TBPS-mSigLIP.git
```

### 1. Download the datasets

```bash
./setup.sh
```

### 2. Install the `uv` package manager and sync dependencies

```bash
cd PERSON_RLF
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 3. Download the `siglip-base-patch16-256-multilingual` checkpoints

```bash
uv run prepare_checkpoints.py
```

### 4. Place the CUHK-FULL dataset in the root directory

Sample project structure:

```bash
.
|-- clip_checkpoints
|-- config
|-- CUHK-PEDES          # Dataset folder for CUHK-PEDES
|-- VN3K                # Dataset folder for VN3K
|-- data
|-- experiments
|-- lightning_data.py
|-- lightning_models.py
|-- model
|-- m_siglip_checkpoints
|-- outputs
|-- prepare_checkpoints.py
|-- __pycache__
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- run.sh
|-- siglip_checkpoints
|-- solver
|-- trainer.py
|-- utils
|-- uv.lock
|-- ...
```

### 5. Log in to Weights & Biases

```bash
uv run wandb login <API_KEY>
```

---

## Running Experiments

### 1. CUHK-FULL dataset

```bash
# With m-SigLIP
# Train with the TBPS method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=random loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=tbps_clip_no_decay optimizer.param_groups.default.lr=1e-5

# Train with the IRRA method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=identity dataset.num_instance=1 loss=irra loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=irra_no_decay optimizer.param_groups.default.lr=1e-5

# Run Circle Loss with LoRA fine-tuning
./run_cir_loss.sh

# Run Circle Loss with full fine-tuning
./run_cir_full.sh
```

---

