import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw
from CFA import CFA
import animeface


# param
num_landmark = 24
img_width = 128
checkpoint_name = 'checkpoint_landmark_191116.pth.tar'
input_img_name = 'test.png'

# detector
face_detector = cv2.CascadeClassifier('lbpcascade_animeface.xml')
landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name=checkpoint_name).cuda()

# transform
normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
train_transform = [transforms.ToTensor(), normalize]
train_transform = transforms.Compose(train_transform)

# input image & detect face
img = cv2.imread(input_img_name)
faces = face_detector.detectMultiScale(img)
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(img)

for x_, y_, w_, h_ in faces:

    # adjust face size
    x = max(x_ - w_ / 8, 0)
    rx = min(x_ + w_ * 9 / 8, img.width)
    y = max(y_ - h_ / 4, 0)
    by = y_ + h_
    w = rx - x
    h = by - y

    # draw result of face detection
    draw.rectangle((x, y, x + w, y + h), outline=(0, 0, 255), width=3)

    # transform image
    img_tmp = img.crop((x, y, x+w, y+h))
    img_tmp = img_tmp.resize((img_width, img_width), Image.BICUBIC)
    img_tmp = train_transform(img_tmp)
    img_tmp = img_tmp.unsqueeze(0).cuda()

    # estimate heatmap
    heatmaps = landmark_detector(img_tmp)
    heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

    # calculate landmark position
    for i in range(num_landmark):
        heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        landmark_y = landmark[0] * h / img_width
        landmark_x = landmark[1] * w / img_width

        # draw landmarks
        draw.ellipse((x + landmark_x - 2, y + landmark_y - 2, x + landmark_x + 2, y + landmark_y + 2), fill=(255, 0, 0))
    
# output image
img.save('output.bmp')
