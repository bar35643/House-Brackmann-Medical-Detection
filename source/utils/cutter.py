#TODO Docstring
"""
TODO
"""

#import statistics
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt


import face_alignment

from .config import LOGGER



class Cutter():
    """
    Check installed dependencies meet requirements

    :param requirements:  List of all Requirements needed (parse *.txt file or list of packages)
    :param exclude: List of all Requirements which will be excuded from the checking (list of packages)
    :param install: True for attempting auto update or False for manual use (True or False)
    """
    def __init__(self, device='cpu', prefix_for_log=""):
        """
        Initializes Cutter Class and Face Alignment module

        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        self.device = device
        self.prefix_for_log = prefix_for_log

        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        self.fn_landmarks = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                         flip_input=False,
                                                         device=str(self.device),
                                                         face_detector='sfd',
                                                         network_size=4)

    def generate_marker(self, img):
        """
        Generates the Landmarks from given input (Points [x, y])

        :param img: input image (Image)
        :returns  Array of Landmarks ([x, y], x and y from type int)
        """
        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        landmarks = self.fn_landmarks.get_landmarks(np.array(img))
        return landmarks[0]

    def cut_symmetry(self, path):
        """
        Cutter Module for the Symmetry. Cropping the input image to the Specs.

        :param path: input image (Image)
        :returns  cropped image
        """
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
        """
        Cutter Module for the Eye. Cropping the input image to the Specs.

        :param path: input image (Image)
        :returns  cropped image
        """
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
        """
        Cutter Module for the Mouth. Cropping the input image to the Specs.

        :param path: input image (Image)
        :returns  cropped image
        """
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
        """
        Cutter Module for the Forehead. Cropping the input image to the Specs.

        :param path: input image (Image)
        :returns  cropped image
        """
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
