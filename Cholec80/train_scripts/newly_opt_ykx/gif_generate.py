from PIL import Image
import imageio
import os

dirp = '/memory/yangkaixiang/SurgicalSceneUnderstanding/pitfalls_bn/data/frames_1fps/41'
frames = []
for i in range(2656, 2672):
    frames.append(Image.open(f"{dirp}/0000{i}.jpg"))

imageio.mimsave('/memory/yangkaixiang/SurgicalSceneUnderstanding/pitfalls_bn/output/predictions/phase/20240717-2115_LongShortV2_kl_cuhknotestSplit_lstm_convnextv2_lr1e-05_bs1_seq64_e2e/41Phase5.gif', frames, duration=5)