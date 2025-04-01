import cv2
import os

dirp = '.../data/frames_1fps/41'
outname = '.../41Phase5.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_fps = 8  # 每秒帧数
video_size = (854, 480)  # 图片尺寸
out = cv2.VideoWriter(outname, fourcc, video_fps, video_size)

for i in range(2656, 2800):
    img = cv2.imread(f"{dirp}/0000{i}.jpg")
    out.write(img)

out.release()