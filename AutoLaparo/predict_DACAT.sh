export CUDA_VISIBLE_DEVICES=0

cd .../AutoLaparo/train_scripts

### CoStoDet-DDPM
python3 save_predictions_DDPM_unslide_LSTMObs_DACAT.py phase \
    --split cuhk1007 \
    --backbone convnext --seq_len 1 --batch_size 1 --DDIM \
    --n_obs_steps 1 --n_action_steps 1 --infer_steps 100 --Obs LSTM \
    --CNN_output_loss \
    --resume  .../CoStoDet-DDPM_AutoLaparo_ConvNeXtv1.pth.tar \