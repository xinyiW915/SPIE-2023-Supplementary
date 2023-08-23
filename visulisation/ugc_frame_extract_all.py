import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.models as models

# open the video
cap = cv2.VideoCapture("./video_sampled_frame/TelevisionClip_1080P-68c6.mkv")  # Get the video object
isOpened = cap.isOpened  # Determine if it is open
# Video information acquisition
fps = cap.get(cv2.CAP_PROP_FPS)

imageNum = 0
frameCount = 0
timef = 30  # Save a picture every 30 frames

image_folder = './video_sampled_frame/'
images = []  # List to store extracted image file paths

while isOpened:

    frameCount += 1
    (frameState, frame) = cap.read()  # Recording of each frame and acquisition status

    if frameState == True and (frameCount % timef == 0):

        # Format transformation, BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to Image
        frame = Image.fromarray(np.uint8(frame))

        imageNum += 1
        fileName = f'{image_folder}{imageNum}.png'  # Temporary storage path
        frame.save(fileName)
        print(fileName + " successfully written")  # Output storage status
        images.append(fileName)

    elif frameState == False:
        break

print('Complete the extraction of video frames!')

cap.release()

# Load images from file paths
loaded_images = []
for image_path in images:
    image = Image.open(image_path)
    loaded_images.append(image)

# Calculate canvas size
num_per_row = 4  # Number of images per row
num_rows = (len(loaded_images) + num_per_row - 1) // num_per_row  # Number of rows
max_width = max(image.width for image in loaded_images)  # Maximum width of an image
max_height = max(image.height for image in loaded_images)  # Maximum height of an image
canvas_width = num_per_row * max_width
canvas_height = num_rows * max_height

# Create a new canvas
canvas = Image.new('RGB', (canvas_width, canvas_height))

# Create a draw object
draw = ImageDraw.Draw(canvas)

# Paste images onto the canvas and add coordinates
x_offset = 0
y_offset = 0
for i, image in enumerate(loaded_images):
    resized_image = image.resize((max_width, max_height))  # Resize the image
    canvas.paste(resized_image, (x_offset, y_offset))
    draw.text((x_offset, y_offset), f"{i+1}", fill=(255, 255, 255))
    x_offset += max_width
    if (i+1) % num_per_row == 0:
        x_offset = 0
        y_offset += max_height

# Save the final image with coordinates
combined_image_path = f'{image_folder}combined_image.png'
canvas.save(combined_image_path)
print('Combined image with coordinates saved successfully!')
