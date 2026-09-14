uv run LoopUS-train \
    --from-hub Thrillcrazyer/Qwen3_1.7B_LoopUS \
    --lma-window 3 --lma-heads 8 \
    --checkpoint-dir checkpoints/LoopUS_LMA3 \
    --train-dataset HuggingFaceFW/fineweb-edu \
    --train-config CC-MAIN-2025-26 --train-split train --train-max-tokens 100000000 \
    --batch-size 2 --gradient-accumulation-steps 5 \
    --learning-rate 5e-5 --max-length 1024 --n-supervision 5 --n-reasoning-steps 20 \
    --encoder-layers 0..1 --decoder-layers 27..27 --log-interval 10 --warmup-steps 20 \
    --eval-interval 0 --wandb