# 2025.4.1 kxyang Cholec80 Recognition Training: BNPifalls; BNPifalls+DDPM; CoStoDet-DDPM

export CUDA_VISIBLE_DEVICES=0

cd .../Cholec80/train_scripts

### BNPifalls
# python3 train.py phase \
#     --split cuhk4040 \
#     --trial_name BNPitfall4040 \
#     --backbone convnext --freeze --workers 4 --seq_len 256 --lr 1e-4


### BNPitfalls + DDPM 
# python3 train_phase_DDPM.py phase \
#     --split cuhk4040 \
#     --trial_name BNPitfalls-DDPM \
#     --backbone convnext --workers 4 --freeze --epochs 200 --random_seed \
#     --seq_len 256 --batch_size 1 --DDIM \
#     --n_obs_steps 256 --n_action_steps 1 --infer_steps 4 --Obs LSTM --lr 1e-3 --weight_decay 1e-2 \
#     --use_ema --CNN_output_loss 

### CoStoDet-DDPM
python3 train_phase_DDPM_DACAT.py phase \
    --split cuhk4040 \
    --trial_name CoStoDet-DDPM \
    --backbone convnextv2 --workers 4 --epochs 50 --random_seed \
    --seq_len 64 --batch_size 1 --DDIM \
    --n_obs_steps 64 --n_action_steps 1 --infer_steps 4 --Obs LSTM --lr 1e-5 --weight_decay 1e-2 \
    --CNN_output_loss 