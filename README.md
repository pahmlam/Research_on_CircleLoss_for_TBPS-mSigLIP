
---
# Optimizing Text-Based Person Search (TBPS) with Circle Loss & mSigLIP
Circle Loss's reference : https://arxiv.org/abs/2002.10857

Resource: 7th Gen Intel(R) Core(TM) i5-7600K GeForce RTX 3060 12GB- 24GB RAM.

Dự án này tập trung tối ưu hóa hệ thống tìm kiếm người dựa trên mô tả văn bản (Text-Based Person Search - TBPS) cho dữ liệu tiếng Việt. Giải pháp cốt lõi là tích hợp **Circle Loss** vào kiến trúc **mSigLIP** để giải quyết các vấn đề về khoảng cách đa phương thức (Cross-modal Gap) và khoảng cách đa ngôn ngữ (Cross-lingual Gap).

## 1. Vấn đề & Động lực
Mô hình TBPS-mSigLIP hiện tại sử dụng hàm mất mát **N-ITC** (Normalized Image-Text Contrastive). Tuy nhiên, N-ITC bộc lộ hạn chế khi xử lý dữ liệu tiếng Việt:
* **Hard Negatives (FP):** Nhầm lẫn giữa những người có ngoại hình tương tự (ví dụ: cùng mặc áo trắng, quần đen).
* **Hard Positives (FN):** Bỏ sót đúng người cần tìm do góc chụp lạ hoặc mô tả văn bản không khớp chặt chẽ.
* **Cơ chế phạt:** N-ITC phạt đều (equal penalty) cho các mẫu dễ và khó, dẫn đến lãng phí tài nguyên tính toán vào các mẫu dễ thay vì tập trung "đào" (mining) các mẫu khó.

## 2. Giải pháp: Circle Loss
Tích hợp **Circle Loss** để thay thế cơ chế phạt đều bằng tối ưu hóa có trọng số thích nghi (adaptive re-weighting):
* **Cơ chế:** Tự động gán trọng số lớn hơn cho các cặp mẫu khó (Hard mining) và trọng số nhỏ hơn cho các mẫu đã học tốt.
* **Mục tiêu:** Tối đa hóa khoảng cách giữa cặp dương ($s_p \to 1$) và cặp âm ($s_n \to 0$) trong không gian embedding.
* **Công thức:**
    $$\mathcal{L}_{circle} = \log[1 + \sum \exp(\gamma \alpha_n s_n) \sum \exp(-\gamma \alpha_p s_p)]$$

## 3. Các chiến lược triển khai (Implementation Strategies)
Dự án đề xuất thử nghiệm 4 chiến lược tích hợp Circle Loss khác nhau (được định nghĩa trong `tbps.py` và `objectives.py`):

### Strategy 1: In-modal Auxiliary Optimization (`strategy="auxiliary"`)
* Sử dụng Circle Loss như một bộ điều chuẩn (regularizer) nội tại cho riêng nhánh ảnh và riêng nhánh văn bản.
* **Mục đích:** Gom cụm các mẫu cùng ID trong từng không gian đơn phương thức trước khi so khớp chéo.

### Strategy 2: Intrinsic N-ITC / NC-ITC (`strategy="intrinsic"`)
* Thay đổi cách tính logit của hàm Sigmoid trong N-ITC bằng cơ chế re-weighting của Circle Loss.
* **Mục đích:** Tối ưu trực tiếp vào ranh giới quyết định (decision boundary).

### Strategy 3: Pure Cross-Modal Circle Loss (`strategy="circle_only"`)
* Thay thế hoàn toàn N-ITC bằng Cross-Modal Circle Loss.
* **Mục đích:** Tạo ra embedding có tính phân tách cao (high separability).

### Strategy 4: Hybrid N-ITC + Cross-Modal Auxiliary (Selected) (`strategy="auxiliary_cross"`)
* **Phương pháp:** Kết hợp song song:
    1.  **Luồng chính (Main):** Dùng N-ITC để duy trì sự ổn định tổng thể.
    2.  **Luồng bổ trợ (Aux):** Dùng Cross-Modal Circle Loss để chuyên "săn" các mẫu khó (hard negatives).
* Áp dụng Multi-View Supervision (MVS) cho cả hai luồng.
* **Kết quả:** Đây là chiến lược hiệu quả nhất được lựa chọn cho thực nghiệm cuối cùng.

## 4. Cấu trúc mã nguồn
* **`model/objectives.py`**: Chứa các hàm tính toán Loss.
    * `compute_cir`: Pairwise Circle Loss cơ bản.
    * `compute_intrinsic_nitc`: Phiên bản Intrinsic tích hợp vào N-ITC.
    * `compute_cross_modal_circle`: Circle Loss áp dụng cho ma trận tương đồng Ảnh-Văn bản.
* **`tbps.py`**: Class chính của mô hình.
    * Quản lý luồng forward và lựa chọn chiến lược loss dựa trên config (`auxiliary`, `intrinsic`, `circle_only`, `auxiliary_cross`).
    * Tích hợp các kỹ thuật khác: LoRA, SS (Self-Supervision), MVS.

## 5. Kết quả thực nghiệm (VN3K Dataset)
Với chiến lược số 4 (Hybrid), mô hình đã đạt được sự cải thiện so với baseline trên tập dữ liệu tiếng Việt:

Test Results:

`+------+-------+-------+-------+-------+-------+`

`| Task |   R1  |   R5  |  R10  |  mAP  |  mINP |`

`+------+-------+-------+-------+-------+-------+`

`| t2i  | 50.53 | 77.78 | 86.43 | 55.94 | 49.37 |`

`| i2t  | 52.35 | 77.55 | 86.20 | 48.97 | 33.27 |`

`+------+-------+-------+-------+-------+-------+`

* **Rank@1:** +0.83%
* **Rank@5:** +1.85%
* **Rank@10:** +1.68%
* **mAP:** +0.98%
* **mINP:** +0.71%

Cấu hình Circle Loss: `Margin = 0.25`, `Gamma = 128`, kết hợp Fine-tune LoRA.

## Setup

0. Clone the repository
```bash
git clone https://github.com/pahmlam/Research_on_CircleLoss_for_TBPS-mSigLIP.git
```

1. Get the datasets
```bash
./setup.sh
```

2. Install uv package manager and sync the dependencies
```bash
cd PERSON_RLF
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

3. Download the `siglip-base-patch16-256-multilingual` checkpoints
```bash
uv run prepare_checkpoints.py
```

4. Put the CUHK-FULL dataset in the root folder
Here is the sample structure of the project
```bash
.
|-- clip_checkpoints
|-- config
|-- CUHK-PEDES          # This is the dataset folder for CUHK-PEDES
|-- VN3K                # This is the dataset folder for VN3K
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

5. Log in to the Weights & Biases
```bash
uv run wandb login <API_KEY>
```

### Run the experiments

1. CUHK-FULL dataset
```bash
# With m-SigLIP
# Run the training with TBPS method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=random loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=tbps_clip_no_decay optimizer.param_groups.default.lr=1e-5
# Run the training with IRRA method
uv run trainer.py -cn m_siglip img_size_str="'(256,256)'" dataset=cuhk_pedes dataset.sampler=identity dataset.num_instance=1 loss=irra loss.softlabel_ratio=0.0 trainer.max_epochs=60 optimizer=irra_no_decay optimizer.param_groups.default.lr=1e-5

# Run circle loss with LoRa
./run_cir_loss.sh

# Run circle loss with full Fine-tune
./run_cir_full.sh
```
