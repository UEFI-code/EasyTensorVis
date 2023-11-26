import math
import cv2
import numpy as np
import torch

def vis1D(tensor, drawSize = 1024, drawline=False, colorful=False):
    # prepare 1024 x 1024 image
    img = np.zeros((drawSize + 1, drawSize + 1, 3), np.uint8)

    # check for max value in tensor and normalize
    max = torch.max(abs(tensor))
    print('max value in tensor: ', max)
    tensor = tensor / max

    # draw circle
    th = 0
    index = 0
    while th < math.pi * 2:
        cosVal, sinVal = math.cos(th), math.sin(th)
        x_0 = int(cosVal * (drawSize / 4) + (drawSize / 2))
        y_0 = int(sinVal * (drawSize / 4) + (drawSize / 2))
        img[x_0, y_0] = (0, 255, 0)
        x_1 = int(cosVal * (drawSize / 2) + (drawSize / 2))
        y_1 = int(sinVal * (drawSize / 2) + (drawSize / 2))
        img[x_1, y_1] = (0, 255, 255)
        x_t = int(cosVal * (drawSize / 4) * tensor[index]) + x_0
        y_t = int(sinVal * (drawSize / 4) * tensor[index]) + y_0
        img[x_t, y_t] = (0, 0, 255)
        if drawline:
            if colorful:
                #cv2.line(img, (x_0, y_0), (x_t, y_t), torch.randint(0, 255, (3,)).tolist(), 1)
                cv2.line(img, (x_0, y_0), (x_t, y_t), calcLineColor(tensor[index]), 1)
            else:
                cv2.line(img, (x_0, y_0), (x_t, y_t), (0, 0, 255), 1)
        th += (math.pi * 2) / tensor.shape[0]
        index += 1
        if index >= tensor.shape[0]:
            break
    return img

def calcLineColor(value):
    # Generate color based on value, like the light colorbar
    hueA = value * 270
    hsv_color = np.array([[[hueA, hueA + 135, 1]]], dtype=np.float32)
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
    rgb_color = rgb_color[0][0] * 255
    return rgb_color.astype(np.uint8).tolist()

if __name__ == '__main__':
    example = torch.randn(512)
    visimg = vis1D(example, drawline=True, colorful=True)
    cv2.imshow('vis', visimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('vis1D.png', visimg)