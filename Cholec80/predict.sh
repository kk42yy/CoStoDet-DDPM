# 2025.4.1 kxyang Cholec80 Recognition Inference: BNPifalls; BNPifalls+DDPM; CoStoDet-DDPM

export CUDA_VISIBLE_DEVICES=0

cd .../Cholec80/train_scripts

### BNPifalls
# python3 save_predictions.py phase \
#     --split cuhk4040 \
#     --backbone convnext --seq_len 64 \
#     --resume .../BNPitfalls_Cholec80_Recognition.pth.tar

### BNPitfalls + DDPM 
# python3 save_predictions_DDPM_unslide_LSTMObs.py phase \
#     --split cuhk4040 \
#     --backbone convnext --seq_len 1 --batch_size 1 --DDIM \
#     --n_obs_steps 1 --n_action_steps 1 --infer_steps 16 --Obs LSTM --use_ema \
#     --CNN_output_loss \
#     --resume .../BNPitfalls-DDPM_Cholec80_Recognition.pth.tar

### CoStoDet-DDPM
python3 save_predictions_DDPM_unslide_LSTMObs_DACAT.py phase \
    --split cuhk4040 \
    --backbone convnextv2 --seq_len 1 --batch_size 1 --DDIM \
    --n_obs_steps 1 --n_action_steps 1 --infer_steps 16 --Obs LSTM \
    --CNN_output_loss \
    --resume .../CoStoDet-DDPM_Cholec80_Recognition.pth.ta