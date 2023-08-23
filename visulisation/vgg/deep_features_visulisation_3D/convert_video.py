import cv2
import os

# 图像文件夹路径
image_folder = '/Users/xxxyy/PycharmProjects/UoB/visulisation/vgg/deep_features_visulisation_3D'

# 输出视频文件路径
output_video_path = '/Users/xxxyy/PycharmProjects/UoB/visulisation/vgg/deep_features_visulisation_3D/vgg16.mp4'

# 获取图像文件列表
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# 读取第一张图像，以获取图像尺寸信息
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, channels = frame.shape

# 创建视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 1.0, (width, height))

# 逐帧将图像写入视频
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(image_path)

    # 重复写入当前帧，以达到停留1秒的效果
    for _ in range(2):  # 每秒30帧，总计重复30次
        video_writer.write(frame)

# 释放视频编码器
video_writer.release()

print('Video successfully created.')
