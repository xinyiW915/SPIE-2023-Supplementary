import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

# Load the pre-trained deep learning model ResNet-50
resnet = models.resnet50(pretrained=True)
# set evaluation mode
resnet.eval()

# Define the input image preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# open the video dir
videos_dir = './video_sampled_frame/'
video_names = 'TelevisionClip_1080P-68c6.mkv'

print(f'Test videos: {video_names}')

video = videos_dir + video_names
print(video + " successfully write in")  # Output storage status
# print(video)

# open the video
cap = cv2.VideoCapture(video)  # Get the video object
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
        print(imageNum)

        fileName = './video_sampled_frame/' + str(imageNum) + '.png'  # Temporary storage path
        cv2.imwrite(fileName, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Load and preprocess the input image
        image_path = f'./video_sampled_frame/{str(imageNum)}.png'
        image = Image.open(image_path)
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch.shape

        # extract features
        with torch.no_grad():
            deep_features = resnet.avgpool(
                resnet.layer4(resnet.relu(resnet.layer3(resnet.layer2(resnet.layer1(resnet.conv1(input_batch)))))))
            deep_features = deep_features.view(deep_features.size(0), -1).numpy()

        print(deep_features)
        print(deep_features.shape)
        np.save(f'./feats/features_deep_resnet50/{str(imageNum)}_deep_features.npy', deep_features)

        # Clear temp data
        # os.remove(fileName)

    elif frameState == False:
        break

print('Complete the extraction of video frames!')
