import numpy as np
import cv2
import torch
from CFA import CFA

# param
num_landmark = 24
img_width = 128
checkpoint_name = 'checkpoint_landmark_191109.pth.tar'
input_img_name = 'test.png'

# detector
face_detector = cv2.CascadeClassifier('lbpcascade_animeface.xml')
landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name=checkpoint_name).cuda()

# input image
img = cv2.imread(input_img_name)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray_img)

for x, y, w, h in faces:

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    # transform image
    crop_img = img[y:y+h, x:x+h]
    crop_img = cv2.resize(crop_img, (img_width, img_width))
    torch_img = crop_img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    torch_img = torch_img.astype('float32') / 255.0
    torch_img = torch.from_numpy(torch_img).cuda()

    # estimate heatmap
    heatmaps = landmark_detector(torch_img)
    heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

    # calculate landmark position
    for i in range(num_landmark):
        heatmaps_tmp = cv2.resize(heatmaps[i], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        landmark_y = landmark[0] * h / img_width
        landmark_x = landmark[1] * w / img_width
        cv2.circle(img, (int(x + landmark_x), int(y + landmark_y)), 3, (0, 0, 255), thickness=-1)
    
# output image
cv2.imwrite('output.bmp', img)
