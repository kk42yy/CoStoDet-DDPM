# 2025.4.1 kxyang Cholec80 Anticipation Training: BNPifalls; CoStoDet-DDPM

export CUDA_VISIBLE_DEVICES=0

cd .../Cholec80/train_scripts_anti

### BNPitfalls
# python3 train.py anticipation \
#     --horizon 5 \
#     --split cuhk6020 \
#     --trial_name BNPitfalls \
#     --backbone convnext --shuffle --epochs 300 --random_seed --CHE --num_ins 12 --seq_len 32 # actual 64


# CoStoDet-DDPM
python3 train_anti_DDPM.py anticipation \
    --horizon 5 \
    --split cuhk6020 \
    --trial_name CoStoDet-DDPM \
    --backbone convnext --shuffle --epochs 300 --random_seed --CHE --seq_len 32 --pre_len 32 --batch_size 1 --DDIM \
    --n_obs_steps 32 --n_action_steps 1 --infer_steps 4 --Obs LSTM --lr 1e-4 --weight_decay 1e-6 --num_ins 12 \
    --use_ema --CNN_output_loss