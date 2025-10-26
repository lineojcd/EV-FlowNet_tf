import os
from pathlib import Path
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


# Define the source directory
result_folder = "test_output"
flow_instance = "IMG_4372_480P_wood"

src_dir = os.path.join(result_folder, flow_instance)

# Initialize lists for each prefix
count_images = []
flow_images = []
timestamp_images = []
grayscale_images = []

# Read files from the directory
if os.path.exists(src_dir):
    files = sorted(os.listdir(src_dir))
    
    for file in files:
        if file.startswith("Count"):
            count_images.append(file)
        elif file.startswith("flow"):
            flow_images.append(file)
        elif file.startswith("Timestamp"):
            timestamp_images.append(file)
        elif file.startswith("Grayscale"):
            grayscale_images.append(file)

# Print results
print(f"Count images: {len(count_images)}")
print(f"Flow images: {len(flow_images)}")
print(f"Timestamp images: {len(timestamp_images)}")
print(f"Grayscale images: {len(grayscale_images)}")

print("Count images sample:", count_images[:3])
print("Flow images sample:", flow_images[:3])
print("Timestamp images sample:", timestamp_images[:3])
print("Grayscale images sample:", grayscale_images[:3])

# Load all images
def load_images(file_list, src_dir):
    images = []
    for file in file_list:
        img = Image.open(os.path.join(src_dir, file))
        images.append(np.array(img))
    return images

count_imgs = load_images(count_images, src_dir)
flow_imgs = load_images(flow_images, src_dir)
timestamp_imgs = load_images(timestamp_images, src_dir)
grayscale_imgs = load_images(grayscale_images, src_dir)

# Get dimensions and create frames
frames = []
height, width = count_imgs[0].shape[:2]

for i in range(len(count_images)):
    # Resize all images to same size
    top_left = cv2.resize(count_imgs[i], (width, height))
    top_right = cv2.resize(timestamp_imgs[i], (width, height))
    bottom_left = cv2.resize(grayscale_imgs[i], (width, height))
    bottom_right = cv2.resize(flow_imgs[i], (width, height))
    
    # Ensure all are 3-channel for concatenation
    if len(top_left.shape) == 2:
        top_left = cv2.cvtColor(top_left, cv2.COLOR_GRAY2BGR)
    if len(top_right.shape) == 2:
        top_right = cv2.cvtColor(top_right, cv2.COLOR_GRAY2BGR)
    if len(bottom_left.shape) == 2:
        bottom_left = cv2.cvtColor(bottom_left, cv2.COLOR_GRAY2BGR)
    if len(bottom_right.shape) == 2:
        bottom_right = cv2.cvtColor(bottom_right, cv2.COLOR_GRAY2BGR)
    
    # Concatenate into 2x2 grid
    top = np.concatenate([top_left, top_right], axis=1)
    bottom = np.concatenate([bottom_left, bottom_right], axis=1)
    frame = np.concatenate([top, bottom], axis=0)
    
    frames.append(frame)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create and save video
# frames: list of numpy arrays (H,W,3) uint8, all same size
clip = ImageSequenceClip(frames, fps=15)   # pass fps here
clip.write_videofile(os.path.join(result_folder,flow_instance+".mp4"), fps=30, codec="libx264", audio=False)
