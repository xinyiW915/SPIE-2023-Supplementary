import cv2
from PIL import Image
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
sum = 0
timef = 30  # Save a picture every 30 frames

sum_samp = 0
while (isOpened):

    sum += 1
    (frameState, frame) = cap.read()  # Recording of each frame and acquisition status

    if frameState == True and (sum % timef == 0):

        # Format transformation, BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to Image
        frame = Image.fromarray(np.uint8(frame))
        frame = np.array(frame)
        # RGBtoBGR meets the opencv display format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        imageNum = imageNum + 1
        fileName = './video_sampled_frame/' +str(imageNum) + '.png'  # Temporary storage path
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in")  # Output storage status
        image = Image.open(fileName)
        # new_image = image.resize((224,224))
        image.save(fileName)

    elif frameState == False:
        break

print('Complete the extraction of video frames!')

cap.release()

