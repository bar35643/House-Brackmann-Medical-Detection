#TODO Docstring
"""
TODO
"""

#import statistics
from copy import deepcopy
import numpy as np
from PIL import Image, ImageOps
#import matplotlib.pyplot as plt


import face_alignment

from .config import LOGGER
from .templates import house_brackmann_template #pylint: disable=import-error



class Cutter():
    """
    Check installed dependencies meet requirements

    :param requirements:  List of all Requirements needed (parse *.txt file or list of packages)
    :param exclude: List of all Requirements which will be excuded from the checking (list of packages)
    :param install: True for attempting auto update or False for manual use (True or False)
    """
    #TODO Redo all functions (input=img_symmetry)
    def __init__(self, device='cpu', prefix_for_log=""):
        """
        Initializes Cutter Class and Face Alignment module

        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        self.device = device
        self.prefix_for_log = prefix_for_log

        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        LOGGER.debug("%sSetting up Framework for generating the Markers", prefix_for_log)
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

    def flip_image_and_return_landmarks(self, img_input):
        """
        Flip images to correct Rotation

        :param image: input Image (Image)
        :returns  landmarks and cropped image (array, Image)
        """
         #try to flip image if exif tag is available see https://pillow.readthedocs.io/en/stable/reference/ImageOps.html
        img_input = ImageOps.exif_transpose(img_input)
        #Generate Landmarks
        det = self.generate_marker(img_input)

        #only executes if rotation with exif tags did not work or rotation is wrong
        #checks rotation from two marker in face and their relative position
        exit_condition = 0
        while not ((det[0, 0] < det[10, 0]) and (det[0, 1] < det[10, 1])):
            if(det[0, 0] < det[10, 0]) and (det[0, 1] > det[10, 1]): #image is turned 90 Degree counterclockwise
                img_input = img_input.transpose(Image.ROTATE_270)
            if(det[0, 0] > det[10, 0]) and (det[0, 1] < det[10, 1]): #image is turned 90 Degree clockwise
                img_input = img_input.transpose(Image.ROTATE_90)
            if(det[0, 0] > det[10, 0]) and (det[0, 1] > det[10, 1]): #image is turned 180 Degree
                img_input = img_input.transpose(Image.ROTATE_180)
            det = self.generate_marker(img_input)

            exit_condition += 1
            assert exit_condition!=10, "Can not turn Images automatically!!"

        return det, img_input


    def load_image(self, path, inv):
        #TODO docstring
        """
        Loading Images

        :param path: Path to image (str)
        :returns  landmarks and cropped image (array, Image)
        """
        img = Image.open(path)
        if inv:
            img = ImageOps.mirror(img)
        dyn_factor = max(int(img.size[0]/1000), int(img.size[1]/1000), 1)

        dyn_factor = dyn_factor+1 if (dyn_factor%2) and not (dyn_factor==1) else dyn_factor
        new_size = (int(img.size[0]/dyn_factor), (int(img.size[1]/dyn_factor)))

        #print(img.size, "to", new_size, "factor", dyn_factor)
        det, img_res = self.flip_image_and_return_landmarks(img.resize(new_size))

        #TODO CROP and return original image (upscale det)

        img_org = img_res
        return det, img_org

    def cut_wrapper(self):
        """
        Function Wrapper

        :returns  Dictionary with the Functions (dict))
        """

        struct_func_list = deepcopy(house_brackmann_template)
        struct_func_list["symmetry"] = self.cut_symmetry
        struct_func_list["eye"] = self.cut_eye
        struct_func_list["mouth"] = self.cut_mouth
        struct_func_list["forehead"] = self.cut_forehead

        return  struct_func_list


    def cut_symmetry(self, path, inv=False):
        """
        Cutter Module for the Symmetry. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        _, img = self.load_image(path, inv)

        # #TODO DELETE
        # plt.imshow(img)
        # plt.scatter(landmarks[:,0], landmarks[:,1],5)
        # plt.scatter(statistics.median(landmarks[:,0]), statistics.median(landmarks[:,1]),10)
        # plt.show()


        return img

    def cut_eye(self, path, inv=False):
        """
        Cutter Module for the Eye. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        landmarks, img = self.load_image(path, inv)
        landmarks = landmarks[slice(36, 48)]

        x_min = landmarks[:,0].min()
        x_max = landmarks[:,0].max()
        y_min = landmarks[:,1].min()
        y_max = landmarks[:,1].max()

        #TODO seperate each eye
        img_slice = img.crop((x_min - 1, y_min - 1, x_max + 1, y_max + 1))

        # #TODO DELETE
        # plt.imshow(img_slice)
        # plt.scatter(landmarks[:,0]-(x_min - 1), landmarks[:,1]-(y_min - 1),5)
        # plt.scatter(statistics.median(landmarks[:,0]-(x_min - 1)), statistics.median(landmarks[:,1]-(y_min - 1)),10)
        # plt.show()


        return img_slice

    def cut_mouth(self, path, inv=False):
        """
        Cutter Module for the Mouth. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        landmarks, img = self.load_image(path, inv)
        landmarks = landmarks[slice(48, 68)]

        x_min = landmarks[:,0].min()
        x_max = landmarks[:,0].max()
        y_min = landmarks[:,1].min()
        y_max = landmarks[:,1].max()

        img_slice = img.crop((x_min - 1, y_min - 1, x_max + 1, y_max + 1))

        # #TODO DELETE
        # plt.imshow(img_slice)
        # plt.scatter(landmarks[:,0]-(x_min - 1), landmarks[:,1]-(y_min - 1),5)
        # plt.scatter(statistics.median(landmarks[:,0]-(x_min - 1)), statistics.median(landmarks[:,1]-(y_min - 1)),10)
        # plt.show()


        return img_slice

    def cut_forehead(self, path, inv=False):
        """
        Cutter Module for the Forehead. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        landmarks, img = self.load_image(path, inv)
        landmarks = landmarks[slice(38, 48)]

        x_min = landmarks[:,0].min()
        x_max = landmarks[:,0].max()
        y_min = landmarks[:,1].min()
        #y_max = landmarks[:,1].max()

        img_slice = img.crop((x_min - 1, 0, x_max + 1, y_min + 1))

        # #TODO DELETE
        # plt.imshow(img_slice)
        # plt.scatter(landmarks[:,0]-(x_min - 1), landmarks[:,1]-(y_min - 1),5)
        # plt.scatter(statistics.median(landmarks[:,0]-(x_min - 1)), statistics.median(landmarks[:,1]-(y_min - 1)),10)
        # plt.show()


        return img_slice
