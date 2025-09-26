export CUDA_VISIBLE_DEVICES=0

cd .../CATARACTS/train_scripts

### BNPifalls (Training feature cache) for CATARACTS
# python3 train.py phase \
# 	--split cuhk2525 \
# 	--trial_name Step1 \
# 	--backbone convnext --freeze --workers 4 --seq_len 256 --lr 5e-4

### CoStoDet-DDPM
python3 train_phase_DDPM_DACAT.py phase \
    --split cuhk2525 --trial_name CoStoDet-DDPM \
    --backbone convnext --workers 4 --epochs 100 --random_seed --seq_len 64 --batch_size 1 --DDIM \
    --n_obs_steps 64 --n_action_steps 1 --infer_steps 4 --Obs LSTM --lr 5e-5 --weight_decay 1e-2 \
    --CNN_output_loss \
    --num_classes 19 --use_ema