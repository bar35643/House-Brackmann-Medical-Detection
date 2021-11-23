
import os
from copy import deepcopy
import itertools

import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from .config import LOGGER
from .templates import house_brackmann_template, house_brackmann_lookup #pylint: disable=import-error



class AverageMeter():
    """Computes and stores the average and sum of values"""
    def __init__(self):
        """
        Initializes AverageMeter Class

        :param avg: Average Value (float)
        :param sum: Sum Value (float)
        :param count: N-Values handled (int)
        """
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """
        Reset all values to 0
        """
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, nun=1):
        """
        update the values
        :param val: Value (float)
        :param num: N times this value (int)
        """
        self.sum += val * nun
        self.count += nun
        self.avg = self.sum / self.count

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



        self.conf_matrix_template = deepcopy(house_brackmann_template)

        for i in self.conf_matrix_template:
            len_enum = len(house_brackmann_lookup[i]["enum"])
            self.conf_matrix_template[i] = np.zeros((len_enum, len_enum)) #https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros/46115998

        self.conf_matrix = {
            "train": deepcopy(self.conf_matrix_template),
            "val": deepcopy(self.conf_matrix_template),
        }
        self.conf_matrix_epoch = {
            "train": deepcopy(self.conf_matrix_template),
            "val": deepcopy(self.conf_matrix_template),
        }



        self.averagemeter = {
            "train": {
                "loss": AverageMeter(),
                "accurancy": AverageMeter()
            },
            "val": {
                "loss": AverageMeter(),
                "accurancy": AverageMeter()
            }
        }

    def reset_averagemeter(self):
        """
        Reset all average Meters to 0
        """
        for i in self.averagemeter:
            for j in self.averagemeter[i]:
                self.averagemeter[i][j] = AverageMeter()

    def update_epoch(self, func):
        """
        Update every epoch

        :param func: function (str)
        """
        print(self.conf_matrix_epoch["train"][func])

        for i in self.conf_matrix_epoch:
            self.conf_matrix_epoch[i] = deepcopy(self.conf_matrix_template)

        #print(self.conf_matrix_epoch["train"][func])

        #Saving as CSV for Each Epoch (Averaged Values)
        fieldnames = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        to_be_saved_dict = {'loss': self.averagemeter["train"]["loss"].avg,
                            'val_loss': self.averagemeter["val"]["loss"].avg,

                            'accuracy': self.averagemeter["train"]["accurancy"].avg,
                            'val_accuracy': self.averagemeter["val"]["accurancy"].avg,}

        filename = os.path.join(self.path, func + ".csv")
        file_exists = os.path.isfile(filename)
        with open(filename, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            if not file_exists:
                writer.writeheader()
            writer.writerow(to_be_saved_dict)


    def update(self, dataset:str, func:str, label, pred, loss):
        """
        Update

        :param dataset: "test" or "val" (str)
        :param func: function (str)
        :param label: True Labels (tensor)
        :param pred: Predicted Labels (tensor)
        :param loss: loss
        """

        #Update Confusion Matrix
        tmp = confusion_matrix(label, pred.argmax(dim=1))
        self.conf_matrix[dataset][func][:tmp.shape[0],:tmp.shape[1]] += tmp
        self.conf_matrix_epoch[dataset][func][:tmp.shape[0],:tmp.shape[1]] += tmp


        #Update AverageMeter
        #use conf_matrix_epoch for recall and ...

        accurancy = accuracy_score(label, pred.max(1)[1].cpu())
        self.averagemeter[dataset]["loss"].update(loss.item())
        self.averagemeter[dataset]["accurancy"].update(accurancy)


        print("loss_avg: ", self.averagemeter[dataset]["loss"].avg,
              "accurancy_avg: ", self.averagemeter[dataset]["accurancy"].avg)






    def plot(self, show=False):
        """
        Creates a new Dictionary with a Preset Value

        :param show: if True show plots  (bool)
        """
        self.confusion_matrix_plot()

        if show:
            plt.show()

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
