#!/bin/bash

echo "========================================================"
echo "CHỌN CHIẾN LƯỢC HUẤN LUYỆN (TRAINING STRATEGY)"
echo "========================================================"
echo "1. Baseline (N-ITC Gốc - Không Circle Loss)"
echo "2. In-modal Auxiliary"
echo "3. Intrinsic (N-ITC biến thể Sigmoid-Circle)"
echo "4. Pure Circle (Thay thế hoàn toàn N-ITC bằng Circle Loss)"
echo "5. Auxiliary Cross-modal (N-ITC + Circle Loss Ảnh-Text)"
echo "========================================================"
read -p "Nhập lựa chọn (1-5): " choice

COMMON_ARGS="
img_size_str='(384,128)'
dataset=vn3k_vi
dataset.sampler=identity
dataset.num_instance=4
dataset.batch_size=24

trainer.max_epochs=60
trainer.accumulate_grad_batches=3
++trainer.precision=16-mixed

optimizer=cir_test
optimizer.param_groups.default.lr=1e-4
+optimizer._target_=torch.optim.AdamW

backbone.freeze.vision=false
backbone.freeze.text=false
+backbone.gradient_checkpointing=true

loss.softlabel_ratio=0.0

"

case $choice in

1)
    echo ">>> Đang chạy MODE 1: BASELINE..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=baseline \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=false \
        loss.circle_loss_weight=0.0 \
        loss.CITC=true
    ;;

2)
    echo ">>> Đang chạy MODE 2: AUXILIARY (Kết hợp)..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=auxiliary \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=true \
        loss.circle_loss_weight=0.1 \
        loss.CITC=true
    ;;

3)
    echo ">>> Đang chạy MODE 3: INTRINSIC (Tích hợp sâu)..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=intrinsic \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=false \
        loss.circle_loss_weight=0.0 \
        loss.CITC=true 
    ;;

4)
    echo ">>> Đang chạy MODE 4: PURE CIRCLE (Thay thế hoàn toàn)..."
    # Pure Circle: Tắt N-ITC chuẩn, dùng Cross-Modal Circle Loss
    # Lưu ý: Vẫn set nitc_loss_weight=1.0 vì trong code ta gán kết quả vào key 'nitc_loss'
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=circle_only \
        loss.NITC=false \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=false \
        loss.circle_loss_weight=0.0 \
        loss.CITC=true 
    ;;

5)
    echo ">>> Đang chạy MODE 5: AUXILIARY CROSS-MODAL..."
    uv run trainer.py -cn cir_msiglip \
        $COMMON_ARGS \
        loss.strategy=auxiliary_cross \
        loss.NITC=true \
        loss.nitc_loss_weight=1.0 \
        loss.CIR=true \
        loss.circle_loss_weight=0.1 \
        loss.CITC=true
    ;;

*)
    echo "Lựa chọn không hợp lệ!"
    exit 1
    ;;
esac