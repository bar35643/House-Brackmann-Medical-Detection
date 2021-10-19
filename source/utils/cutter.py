import statistics
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


import face_alignment

from .config import LOGGER



class Cutter():
    def __init__(self, device='cpu', prefix_for_log=""):
        self.device = device
        self.prefix_for_log = prefix_for_log

        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        self.fn_landmarks = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                         flip_input=False,
                                                         device=str(self.device),
                                                         face_detector='sfd',
                                                         network_size=4)

    def generate_marker(self, img):
        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        landmarks = self.fn_landmarks.get_landmarks(np.array(img))
        return landmarks[0]

    def cut_symmetry(self, path):
        img = Image.open(path)
        landmarks = self.generate_marker(img)

        x_min = landmarks[landmarks[:,0].argmin()]
        x_max = landmarks[landmarks[:,0].argmax()]
        y_min = landmarks[landmarks[:,1].argmin()]
        y_max = landmarks[landmarks[:,1].argmax()]

        img_slice = img.crop((x_min[0] - 1, y_min[1] - 1, x_max[0] + 1, y_max[1] + 1))

        # landmarks[:,0] = landmarks[:,0] - x_min[0]
        # landmarks[:,1] = landmarks[:,1] - y_min[1]
        #
        # # #TODO DELETE
        # # plt.imshow(img_slice)
        # # plt.scatter(landmarks[:,0], landmarks[:,1],5)
        # # plt.scatter(statistics.median(landmarks[:,0]), statistics.median(landmarks[:,1]),10)
        # # plt.scatter(x_min[0], x_min[1],15, color='red')
        # # plt.scatter(x_max[0], x_max[1],15, color='red')
        # # plt.scatter(y_min[0], y_min[1],15, color='red')
        # # plt.scatter(y_max[0], y_max[1],15, color='red')
        # # plt.show()


        return img_slice

    def cut_eye(self, path):
        img = Image.open(path)
        landmarks = self.generate_marker(img)
        landmarks = landmarks[slice(36, 48)]

        x_min = landmarks[landmarks[:,0].argmin()]
        x_max = landmarks[landmarks[:,0].argmax()]
        y_min = landmarks[landmarks[:,1].argmin()]
        y_max = landmarks[landmarks[:,1].argmax()]

        img_slice = img.crop((x_min[0] - 1, y_min[1] - 1, x_max[0] + 1, y_max[1] + 1))

        # landmarks[:,0] = landmarks[:,0] - x_min[0]
        # landmarks[:,1] = landmarks[:,1] - y_min[1]
        #
        # # #TODO DELETE
        # # plt.imshow(img_slice)
        # # plt.scatter(landmarks[:,0], landmarks[:,1],5)
        # # plt.scatter(statistics.median(landmarks[:,0]), statistics.median(landmarks[:,1]),10)
        # # plt.scatter(x_min[0], x_min[1],15, color='red')
        # # plt.scatter(x_max[0], x_max[1],15, color='red')
        # # plt.scatter(y_min[0], y_min[1],15, color='red')
        # # plt.scatter(y_max[0], y_max[1],15, color='red')
        # # plt.show()


        return img_slice

    def cut_mouth(self, path):
        img = Image.open(path)
        landmarks = self.generate_marker(img)
        landmarks = landmarks[slice(48, 68)]

        x_min = landmarks[landmarks[:,0].argmin()]
        x_max = landmarks[landmarks[:,0].argmax()]
        y_min = landmarks[landmarks[:,1].argmin()]
        y_max = landmarks[landmarks[:,1].argmax()]

        img_slice = img.crop((x_min[0] - 1, y_min[1] - 1, x_max[0] + 1, y_max[1] + 1))

        # landmarks[:,0] = landmarks[:,0] - x_min[0]
        # landmarks[:,1] = landmarks[:,1] - y_min[1]
        #
        # # #TODO DELETE
        # # plt.imshow(img_slice)
        # # plt.scatter(landmarks[:,0], landmarks[:,1],5)
        # # plt.scatter(statistics.median(landmarks[:,0]), statistics.median(landmarks[:,1]),10)
        # # plt.scatter(x_min[0], x_min[1],15, color='red')
        # # plt.scatter(x_max[0], x_max[1],15, color='red')
        # # plt.scatter(y_min[0], y_min[1],15, color='red')
        # # plt.scatter(y_max[0], y_max[1],15, color='red')
        # # plt.show()


        return img_slice

    def cut_forehead(self, path):
        img = Image.open(path)
        landmarks = self.generate_marker(img) #TODO

        x_min = landmarks[landmarks[:,0].argmin()]
        x_max = landmarks[landmarks[:,0].argmax()]

        landmarks = landmarks[slice(38, 48)]

        y_min = landmarks[landmarks[:,1].argmin()]

        img_slice = img.crop((x_min[0] - 1, 0, x_max[0] + 1, y_min[1] + 1))

        # landmarks[:,0] = landmarks[:,0] - x_min[0]
        # landmarks[:,1] = landmarks[:,1] - y_min[1]
        #
        # # #TODO DELETE
        # # plt.imshow(img_slice)
        # # plt.scatter(landmarks[:,0], landmarks[:,1],5)
        # # plt.scatter(statistics.median(landmarks[:,0]), statistics.median(landmarks[:,1]),10)
        # # plt.scatter(x_min[0], x_min[1],15, color='red')
        # # plt.scatter(x_max[0], x_max[1],15, color='red')
        # # plt.scatter(y_min[0], y_min[1],15, color='red')
        # # plt.scatter(y_max[0], y_max[1],15, color='red')
        # # plt.show()


        return img_slice
