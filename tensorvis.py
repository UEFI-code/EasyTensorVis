import math
import cv2
import numpy as np

def vis1D(tensor, drawSize = 1024, drawline=False):
    # prepare 1024 x 1024 image
    img = np.zeros((drawSize + 1, drawSize + 1, 3), np.uint8)

    # check for max value in tensor and normalize
    max = torch.max(abs(tensor))
    print('max value in tensor: ', max)
    tensor = tensor / max
    
    # draw circle
    i = 0
    index = 0
    while i < math.pi * 2:
        x_0 = int(math.cos(i) * (drawSize / 4) + (drawSize / 2))
        y_0 = int(math.sin(i) * (drawSize / 4) + (drawSize / 2))
        img[x_0, y_0] = (0, 255, 0)
        x_1 = int(math.cos(i) * (drawSize / 2) + (drawSize / 2))
        y_1 = int(math.sin(i) * (drawSize / 2) + (drawSize / 2))
        img[x_1, y_1] = (0, 255, 255)
        x_t = int(math.cos(i) * (drawSize / 2) * tensor[index] + (drawSize / 2))
        y_t = int(math.sin(i) * (drawSize / 2) * tensor[index] + (drawSize / 2))
        img[x_t, y_t] = (0, 0, 255)
        if drawline:
            cv2.line(img, (x_0, y_0), (x_t, y_t), (0, 0, 255), 1)
        i += (math.pi * 2) / example.shape[0]
        index += 1
    cv2.imshow('img', img)

if __name__ == '__main__':
    import torch
    example = torch.rand(512)
    vis1D(example, drawline=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()