export CUDA_VISIBLE_DEVICES=0

cd .../CATARACTS/train_scripts

### CoStoDet-DDPM
python3 save_predictions_DDPM_unslide_LSTMObs_DACAT.py phase \
    --split cuhk2525 \
    --backbone convnext --seq_len 1 --batch_size 1 --DDIM \
    --n_obs_steps 1 --n_action_steps 1 --infer_steps 16 --Obs LSTM --use_ema \
    --num_classes 19 --CNN_output_loss \
    --resume .../CoStoDet-DDPM_CATARACTS_Recognition.pth.tar \