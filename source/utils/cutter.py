#TODO Docstring
"""
TODO
"""
import sys
import os
from copy import deepcopy
from functools import lru_cache
import numpy as np
from PIL import Image, ImageOps
#import matplotlib.pyplot as plt

import face_alignment

from .config import LOGGER, IMG_FORMATS  #pylint: disable=import-error
from .templates import house_brackmann_template #pylint: disable=import-error
from .singleton import Singleton #pylint: disable=import-error

if sys.platform == 'win32':
    import errorimports as pyheif #pylint: disable=import-error
if sys.platform == 'linux':
    import pyheif #pylint: disable=import-error


def load_images_format(path, img_name):
    """
    Loading images in correct Format

    :param path: Path to folder (str)
    :param img_name: Name of the Image (str)
    :returns  Image (Image)
    """
    path_list = os.listdir(path)
    matching_folders = [os.path.join(path, s) for s in path_list if ("T000" in s) and ("postop" not in s)]

    if not matching_folders:
        matching_folders = [path]

    matching_img_path = []
    for i in matching_folders:
        matching_img_path += [os.path.join(i, s) for s in os.listdir(i)]

    matching_img_path_format = [x for x in matching_img_path if x.split('.')[-1].lower() in IMG_FORMATS and not ("IMG" in x)]
    matching_img_path_name   = [x for x in matching_img_path_format if img_name in x.split('/')[-1]]
    matching_img_path_name.sort()

    # print("------")
    # for j in matching_img_path_format:
    #     print(pic                        , "   ########   ",
    #           (pic in j.split('/')[-1]) , "   ########   ",
    #           j.split('/')[-1]         , "   ########   ",
    #           j.split('.')[-1].lower() , "   ########   ",
    #           j  )
    #
    # print("------")

    image = matching_img_path_name if matching_img_path_name else matching_img_path_format
    assert image, f"Error While loading Image at Path {matching_img_path}"

    if image[0].split('.')[-1].lower() in ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']:
        return Image.open(image[0])

    if image[0].split('.')[-1].lower() in ['heic']:
        temp = pyheif.read_heif(image[0])
        return Image.frombytes(mode=temp.mode, size=temp.size, data=temp.data)

    return None

@Singleton
class Cutter():
    """
    Check installed dependencies meet requirements

    :param requirements:  List of all Requirements needed (parse *.txt file or list of packages)
    :param exclude: List of all Requirements which will be excuded from the checking (list of packages)
    :param install: True for attempting auto update or False for manual use (True or False)
    """
    #TODO Redo all functions (input=img_symmetry)
    def __init__(self):
        """
        Initializes Cutter Class and Face Alignment module

        :param device: cuda device (cpu or cuda:0)
        :param prefix_for_log: logger output prefix (str)
        """
        self.prefix_for_log = ""
        self.fn_landmarks = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                         flip_input=False,
                                                         device='cpu',
                                                         face_detector='sfd',
                                                         network_size=4)

    def set(self, device, prefix_for_log:str):
        """
        set Class items
        :param device: cuda device (cpu or cuda)
        :param prefix_for_log: logger output prefix (str)
        """
        self.prefix_for_log = prefix_for_log
        device = 'cpu' if str(device) == "cpu" else 'cuda'

        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        #pylint: disable=protected-access
        LOGGER.debug("%sSetting up Framework for generating the Markers", self.prefix_for_log)
        self.fn_landmarks = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                         flip_input=False,
                                                         device=device,
                                                         face_detector='sfd',
                                                         network_size=4)

    def generate_marker(self, img):
        """
        Generates the Landmarks from given input (Points [x, y])

        :param img: input image (Image)
        :returns  Array of Landmarks ([x, y], x and y from type int)
        """
        #Documentation for Framework: https://github.com/1adrianb/face-alignment
        assert self.fn_landmarks, "Use Cutter.instanche().set(<properties>) to set the self Values!"
        landmarks = self.fn_landmarks.get_landmarks(np.array(img))
        return landmarks[0]

    def flip_image_and_return_landmarks(self, img_input, dyn_factor):
        """
        Flip images to correct Rotation

        :param image: input Image (Image)
        :returns  landmarks and cropped image (array, Image)
        """
         #try to flip image if exif tag is available see https://pillow.readthedocs.io/en/stable/reference/ImageOps.html
        img_input = ImageOps.exif_transpose(img_input)
        img_resized = img_input.resize((int(img_input.size[0]/dyn_factor), int(img_input.size[1]/dyn_factor)))
        #Generate Landmarks
        det = self.generate_marker(img_resized)

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

            img_resized = img_input.resize((int(img_input.size[0]/dyn_factor), int(img_input.size[1]/dyn_factor)))
            det = self.generate_marker(img_resized)

            exit_condition += 1
            assert exit_condition!=10, "Can not turn Images automatically!!"

        return det, img_input

    def load_image(self, path, img_name):
        """
        Loading Images

        :param path: Path to image (str)
        :returns  landmarks and cropped image (array, Image)
        """



        img = load_images_format(path, img_name)

        dyn_factor = max(int(img.size[0]/1000), int(img.size[1]/1000), 1)
        dyn_factor = dyn_factor+1 if dyn_factor%2 else dyn_factor
        #print(img.size, "to", new_size, "factor", dyn_factor)
        det, img_flip_org = self.flip_image_and_return_landmarks(img, dyn_factor)

        assert len(det), "Marker Detection Failture"

        x_min_det = det[:,0].min()*dyn_factor
        x_max_det = det[:,0].max()*dyn_factor
        y_min_det = det[:,1].min()*dyn_factor
        y_max_det = det[:,1].max()*dyn_factor

        x_diff = abs(x_max_det-x_min_det)
        y_diff = abs(y_max_det-y_min_det)

        x_min = int(x_min_det - (x_diff/8)) if int(x_min_det - (x_diff/8)) > 0 else 0
        x_max = int(x_max_det + (x_diff/8)) if int(x_max_det + (x_diff/8)) < img.size[0] else img.size[0]

        y_min = int(y_min_det - (y_diff/2)) if int(y_min_det - (y_diff/2)) > 0 else 0
        y_max = int(y_max_det + (y_diff/4)) if int(y_max_det + (y_diff/4)) < img.size[1] else img.size[1]

        img_crop = img_flip_org.crop((x_min,y_min,x_max,y_max))
        det = det*dyn_factor - [x_min, y_min]

        return det, img_crop

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

    def cut_symmetry(self, path, img_name):
        """
        Cutter Module for the Symmetry. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        _, img = self.load_image(path, img_name)
        # plt.imshow(img)
        # plt.scatter(landmarks[:,0], landmarks[:,1],10, color=[1, 0, 0, 1])
        # plt.show()
        return img

    def cut_eye(self, path, img_name):
        """
        Cutter Module for the Eye. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        landmarks, img = self.load_image(path, img_name)
        eye = landmarks[slice(36, 42)]
        fac = (landmarks[:,0].min())/4

        x_min_eye = int(eye[:,0].min() -fac)
        x_max_eye = int(eye[:,0].max() +fac)
        y_min_eye = int(eye[:,1].min() -fac)
        y_max_eye = int(eye[:,1].max() +fac)

        img_slice_left = img.crop((x_min_eye,y_min_eye,x_max_eye,y_max_eye))

        eye = landmarks[slice(42, 48)]
        x_min_eye = int(eye[:,0].min() -fac)
        x_max_eye = int(eye[:,0].max() +fac)
        y_min_eye = int(eye[:,1].min() -fac)
        y_max_eye = int(eye[:,1].max() +fac)

        img_slice_right = img.crop((x_min_eye,y_min_eye,x_max_eye,y_max_eye))


        width = img_slice_right.width   if img_slice_right.width  >= img_slice_left.width  else img_slice_left.width
        height = img_slice_right.height if img_slice_right.height >= img_slice_left.height else img_slice_left.height

        img_slice_right = img_slice_right.resize((width,height))
        img_slice_left  =  img_slice_left.resize((width,height))

        dst = Image.new('RGB', (img_slice_left.width + img_slice_right.width, img_slice_left.height))
        dst.paste(img_slice_left, (0, 0))
        dst.paste(img_slice_right, (img_slice_left.width, 0))

        # plt.imshow(dst)
        # plt.show()

        return img_slice_right

    def cut_mouth(self, path, img_name):
        """
        Cutter Module for the Mouth. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        landmarks, img = self.load_image(path, img_name)
        landmarks = landmarks[slice(48, 68)]
        fac = (landmarks[:,0].min())/4

        x_min_mouth = int(landmarks[:,0].min() -fac)
        x_max_mouth = int(landmarks[:,0].max() +fac)
        y_min_mouth = int(landmarks[:,1].min() -fac)
        y_max_mouth = int(landmarks[:,1].max() +fac)

        img_slice = img.crop((x_min_mouth,y_min_mouth,x_max_mouth,y_max_mouth))
        # plt.imshow(img_slice)
        # plt.scatter(landmarks[:,0]-x_min_mouth, landmarks[:,1]-y_min_mouth,10, color=[1, 0, 0, 1])
        # plt.show()
        return img_slice

    def cut_forehead(self, path, img_name):
        """
        Cutter Module for the Forehead. Cropping the input image to the Specs.

        :param path: input path
        :returns  cropped image
        """
        landmarks, img = self.load_image(path, img_name)
        landmarks = landmarks[slice(17, 27)]
        fac = (landmarks[:,0].min())/16

        x_min_forehead = 0
        x_max_forehead = img.size[0]
        y_min_forehead = 0
        y_max_forehead = int(landmarks[:,1].max()+fac)

        img_slice = img.crop((x_min_forehead,y_min_forehead,x_max_forehead,y_max_forehead))
        # plt.imshow(img_slice)
        # plt.scatter(landmarks[:,0]-x_min_forehead, landmarks[:,1]-y_min_forehead,10, color=[1, 0, 0, 1])
        # plt.show()
        return img_slice
