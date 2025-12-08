uv run trainer.py -cn cir_msiglip \
    img_size_str="'(256,256)'" \
    dataset=cuhk_pedes \
    dataset.sampler=random \
    loss.softlabel_ratio=0.0 \
    trainer.max_epochs=60 \
    optimizer=cir_test \
    optimizer.param_groups.default.lr=1e-4 \
    \
    dataset.batch_size=16 \
    trainer.accumulate_grad_batches=4 \
    \
    backbone.freeze.vision=true \
    backbone.freeze.text=true \
    \
    +lora._target_=peft.LoraConfig \
    +lora.r=16 \
    +lora.lora_alpha=32 \
    +lora.lora_dropout=0.1 \
    +lora.bias="none" \
    +lora.target_modules=["q_proj","v_proj","k_proj","out_proj"] \
    +lora.task_type="FEATURE_EXTRACTION" \
    +lora.inference_mode=false \
    +optimizer._target_=torch.optim.AdamW 