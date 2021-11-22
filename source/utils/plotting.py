
import os
from copy import deepcopy
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .config import LOGGER
from .templates import house_brackmann_template, house_brackmann_lookup #pylint: disable=import-error



class Plotting():
    """
    Class for Plotting
    """
    def __init__(self, path="", nosave=False, prefix_for_log=""):
        """
        Initializes the Plotting class
        :param path: Path for saving the images (str)
        :param nosave: If False Plots will be saved (bool)
        """
        self.prefix_for_log = prefix_for_log
        self.path = path
        self.nosave=nosave

        conf_matrix = deepcopy(house_brackmann_template)
        for i in conf_matrix:
            len_enum = len(house_brackmann_lookup[i]["enum"])
            conf_matrix[i] = np.zeros((len_enum, len_enum)) #https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros/46115998

        self.conf_matrix = {
            "train": deepcopy(conf_matrix),
            "val": deepcopy(conf_matrix),
        }

        self.params = {
            "dpi": 500, #higher dpi higher res
            "saveformat": "png", #eps, png, ...
            "fontsize": 16,
            "fontweight": "bold",
            "cmap": "Blues"  #Color for the Heatmap one of: 'Greys', 'Purples', 'Blues', 'Greens',
                             #                              'Oranges', 'Reds','YlOrBr', 'YlOrRd',
                             #                              'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu',
                             #                              'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        }

    def plot(self, show=False):
        """
        Creates a new Dictionary with a Preset Value

        :param show: if True show plots  (bool)
        """
        self.confusion_matrix_plot()

        if show:
            plt.show()

    def confusion_matrix_update(self, dataset:str, func:str, label, pred):
        """
        Update Confusion Martix Values

        :param set: "test" or "val" (str)
        :param func: function (str)
        :param label: True Labels (tensor)
        :param pred: Predicted Labels (tensor)
        """
        tmp = confusion_matrix(label, pred.argmax(dim=1))
        self.conf_matrix[dataset][func][:tmp.shape[0],:tmp.shape[1]] += tmp

    def confusion_matrix_plot(self, normalize=False, title='Confusion Matrix'):
        """
        Creates the Confusion Matrix as Matplotlib

        :param normalize: Normalizes Values (bool)
        :param title: Title of the Matrix (str)

        :return fig
        """
        fig, axs = plt.subplots(len(list(self.conf_matrix.keys())),         #Collum
                                len(list(house_brackmann_lookup.keys())),   #Row
                                figsize=(12, 12))

        if normalize:
            LOGGER.info("%sNormalized confusion matrix", self.prefix_for_log)
        else:
            LOGGER.info("%sConfusion matrix without normalization", self.prefix_for_log)

        for col, dataset in enumerate(list(self.conf_matrix.keys())):
            for row, func in enumerate(list(house_brackmann_lookup.keys())):

                tmp = self.conf_matrix[dataset][func]
                tmp = tmp.astype('int') if not normalize else tmp.astype('float') / tmp.sum(axis=1)[:, np.newaxis]

                axs[col, row].imshow(tmp, interpolation='nearest', cmap=self.params["cmap"])
                axs[col, row].set_title(dataset+"_"+func)
                #axs[col, row].colorbar()

                keys = list(house_brackmann_lookup[func]["enum"].keys())

                axs[col, row].set_xticks(np.arange(len(keys)))
                axs[col, row].set_xticklabels(keys, rotation=45)

                axs[col, row].set_yticks(np.arange(len(keys)))
                axs[col, row].set_yticklabels(keys)

                fmt = '.2f' if normalize else 'd'
                thresh = tmp.max() / 2.
                for i, j in itertools.product(range(tmp.shape[0]), range(tmp.shape[1])):
                    axs[col, row].text(j, i,                                               #Positon
                                       format(tmp[i, j], fmt),                             #Formatted Value
                                       horizontalalignment="center",                       #Alignment
                                       color="white" if tmp[i, j] > thresh else "black")   #color of text

                    axs[col, row].set_ylabel('True label')
                    axs[col, row].set_xlabel('Predicted label')

        #fig.subplots_adjust(wspace=0.2, hspace=0.2)
        fig.suptitle(title, y=0.8, fontsize=self.params["fontsize"], fontweight=self.params["fontweight"])
        fig.tight_layout(rect=[0, 0, 1, 1])
        fig.subplots_adjust(hspace=-0.5)

        if not self.nosave:
            plt.savefig(os.path.join(self.path, "confusion_matrix."+self.params["saveformat"]),
                        format=self.params["saveformat"],
                        dpi=self.params["dpi"],
                        bbox_inches="tight")
        return fig
