"""
# Copyright (c) 2021-2022 Raphael Baumann and Ostbayerische Technische Hochschule Regensburg.
#
# This file is part of house-brackmann-medical-processing
# Author: Raphael Baumann
#
# License:
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Changelog:
# - 2021-12-15 Initial (~Raphael Baumann)
"""

import os
from copy import deepcopy
import itertools

import csv
#import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from .config import LOGGER
from .templates import house_brackmann_template, house_brackmann_lookup #pylint: disable=import-error
from .general import merge_two_dicts


np.seterr(divide='ignore', invalid='ignore')

#https://en.wikipedia.org/wiki/Confusion_matrix
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
#https://deeplizard.com/learn/video/0LhiS6yu2qQ

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
        self.vallist = np.array([])

    def reset(self):
        """
        Reset all values to 0
        """
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vallist = np.array([])

    def update(self, val, nun=1):
        """
        update the values
        :param val: Value (float)
        :param num: N times this value (int)
        """
        self.vallist = np.append(val, self.vallist, axis=None)
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
    def statistics_criteria_calculation(self, dataset, conf_matrix):
        """
        Calculate Statistics
        :param dataset: "train" or "val" (str)
        :param conf_matrix: Confusion Matrix (NxN float)
        :return dict

        Problems:
            - 0/0 = nan
            - a/0 = inf
            - 0/a = 0

        Solution Limit of Function:
            x for Divisor in R >=0
            a for Divident in R >0

            lim(0/x) = 0               Operation 0/0 -> Value =  nan = 0
            x->0
            lim(0/x) = 0               Operation 0/10 -> Value =  0
            x->+/- inf


            lim(a/x) = inf             Operation a/x  -> Value = inf =1 (maximum expected Value)
            x->0
            lim(a/x) = 0               Operation a/x  -> Value 0
            x->+/- inf

            lim(-a/x) = -inf           Operation a/x  -> Value = -inf (not expected, all Values are positive)
            x->0
            lim(-a/x) = 0              Operation -a/x  -> Value 0
            x->+/- inf

        """



        false_positive = (conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        false_negative = (conf_matrix.sum(axis=1) - np.diag(conf_matrix))
        true_positive = (np.diag(conf_matrix))
        true_negative = (conf_matrix.sum() - (false_positive + false_negative + true_positive))

        # Sensitivity, hit rate, recall, or true positive rate
        # Counterpart: miss rate or False negative rate fnr=1-tpr
        tpr = true_positive/(true_positive+false_negative)

        # Specificity, correct rejection rate or true negative rate
        # Counterpart: Fallout or false positive rate fpr=1-tnr
        tnr = true_negative/(true_negative+false_positive)

        # Precision or positive predictive value
        # Counterpart: False discovery rate fdr=1-ppv
        ppv = true_positive/(true_positive+false_positive)

        # Negative predictive value
        # Counterpart: False omission rate for=1-npv
        npv = true_negative/(true_negative+false_negative)

        # accuracy
        acc = (true_positive+true_negative)/(true_positive+false_positive+false_negative+true_negative)

        # F1 Score ( harmonic mean of precision and sensitivity)
        f1_score = (2*true_positive)/(2*true_positive+false_positive+false_negative)

        #TODO LR+, LR-, DOR ??

        #Correction Terms if nan occurs
        tpr       = np.where(np.isnan(tpr)        , 0, tpr)
        tnr       = np.where(np.isnan(tnr)        , 0, tnr)
        ppv       = np.where(np.isnan(ppv)        , 0, ppv)
        npv       = np.where(np.isnan(npv)        , 0, npv)
        acc       = np.where(np.isnan(acc)        , 0, acc)
        f1_score  = np.where(np.isnan(f1_score)   , 0, f1_score)

        #Correction Terms if inf occurs
        tpr       = np.where(np.isposinf(tpr)     , 1, tpr)
        tnr       = np.where(np.isposinf(tnr)     , 1, tnr)
        ppv       = np.where(np.isposinf(ppv)     , 1, ppv)
        npv       = np.where(np.isposinf(npv)     , 1, npv)
        acc       = np.where(np.isposinf(acc)     , 1, acc)
        f1_score  = np.where(np.isposinf(f1_score), 1, f1_score)

        LOGGER.info("%s Confusion Matrix: \n%s", dataset, conf_matrix)
        LOGGER.info("FP: %s --- FN: %s --- TP: %s --- Tp: %s", false_positive, false_negative, true_positive, true_negative   )
        LOGGER.info("Loss: %s -- %s"       , self.averagemeter[dataset]["loss"].vallist, self.averagemeter[dataset]["loss"].vallist.mean())
        LOGGER.info("Sensitivity: %s -- %s", tpr, tpr.mean()             )
        LOGGER.info("Specificity: %s -- %s", tnr, tnr.mean()             )
        LOGGER.info("PPV: %s -- %s"        , ppv, ppv.mean()             )
        LOGGER.info("NPV: %s -- %s"        , npv, npv.mean()             )
        LOGGER.info("F1 Score: %s -- %s"   , f1_score, f1_score.mean()   )
        LOGGER.info("Accurancy: %s -- %s\n", acc, acc.mean()             )


        ret_dict = { dataset+"_loss": self.averagemeter[dataset]["loss"].avg,
                     dataset+"_tpr": tpr.mean(),
                     dataset+"_tnr": tnr.mean(),
                     dataset+"_ppv": ppv.mean(),
                     dataset+"_npv": npv.mean(),
                     dataset+"_f1" : f1_score.mean(),
                     dataset+"_acc": acc.mean()}

        #print(json.dumps(ret_dict, indent = 4))
        return ret_dict

    def update_epoch(self, func):
        """
        Update every epoch

        :param func: function (str)
        """
        train_dict = self.statistics_criteria_calculation("train", self.conf_matrix_epoch["train"][func])
        val_dict = self.statistics_criteria_calculation("val", self.conf_matrix_epoch["val"][func])

        to_be_saved_dict = merge_two_dicts(train_dict, val_dict)
        fieldnames = list(to_be_saved_dict.keys())

        if not self.nosave: #nosave is not enabled
            #Saving as CSV for Each Epoch (Averaged Values)
            filename = os.path.join(self.path, func + ".csv")
            file_exists = os.path.isfile(filename)
            with open(filename, 'a+', newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(to_be_saved_dict)



        #Reset conf_matrix_epoch
        #diable is better for reading
        for i in self.conf_matrix_epoch: #pylint: disable=consider-using-dict-items
            self.conf_matrix_epoch[i][func].fill(0)

        #Reset all AverageMeter
        for i in self.averagemeter: #pylint: disable=consider-using-dict-items
            for j in self.averagemeter[i]:
                self.averagemeter[i][j].reset()

        return val_dict


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
        tmp = confusion_matrix(label, pred.argmax(dim=1), labels=list(house_brackmann_lookup[func]["enum"].values())  )

        self.conf_matrix[dataset][func] += tmp
        self.conf_matrix_epoch[dataset][func] += tmp


        #Update AverageMeter
        accurancy = accuracy_score(label, pred.max(1)[1].cpu())
        self.averagemeter[dataset]["loss"].update(loss.item())
        self.averagemeter[dataset]["accurancy"].update(accurancy)

        LOGGER.info("Average from Batch -- loss-avg=%s, accurancy-avg=%s", self.averagemeter[dataset]["loss"].avg, self.averagemeter[dataset]["accurancy"].avg)



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
                #print(dataset, func,  tmp)
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
                thresh = tmp.max() / 1.2
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
