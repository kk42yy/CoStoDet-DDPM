# 2025.4.1 kxyang Cholec80 Anticipation Inference: BNPifalls; CoStoDet-DDPM

export CUDA_VISIBLE_DEVICES=0

cd .../Cholec80/train_scripts_anti

### BNPitfalls
# python3 save_predictions.py anticipation \
#     --horizon 5 \
#     --split cuhk6020 \
#     --backbone convnext --num_ins 12 \
#     --resume .../5min_model3.pth.tar


# CoStoDet-DDPM
python3 save_predictions_DDPM_unslide_LSTMObs.py anticipation \
    --horizon 5 \
    --split cuhk6020 \
    --backbone convnext --seq_len 1 --batch_size 1 --DDIM --pre_len 1 \
    --n_obs_steps 1 --n_action_steps 1 --infer_steps 16 --Obs LSTM --num_ins 12 \
    --use_ema --CNN_output_loss \
    --resume .../5min_model3.pth.tar \