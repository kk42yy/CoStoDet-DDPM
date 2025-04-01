from PIL import Image
import imageio
import os

dirp = '.../data/frames_1fps/41'
frames = []
for i in range(2656, 2672):
    frames.append(Image.open(f"{dirp}/0000{i}.jpg"))

imageio.mimsave('.../41Phase5.gif', frames, duration=5)