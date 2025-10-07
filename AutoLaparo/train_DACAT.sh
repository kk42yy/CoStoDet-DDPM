export CUDA_VISIBLE_DEVICES=0

cd .../AutoLaparo/train_scripts

### BNPifalls (Training feature cache) for AutoLaparo
# python3 train.py phase \
# 	--split cuhk1007 \
# 	--trial_name Step1 \
# 	--backbone convnextv2 --freeze --workers 4 --seq_len 256 --lr 5e-4

### CoStoDet-DDPM
python3 train_phase_DDPM_DACAT.py phase \
    --split cuhk1007 --trial_name CoStoDet-DDPM \
    --backbone convnextv2 --workers 4 --epochs 50 --random_seed --seq_len 64 --batch_size 1 --DDIM \
    --n_obs_steps 64 --n_action_steps 1 --infer_steps 4 --Obs LSTM --lr 1e-5 --weight_decay 1e-2 \
    --CNN_output_loss \
