
import os
from copy import deepcopy
import itertools

import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from .config import LOGGER
from .templates import house_brackmann_template, house_brackmann_lookup #pylint: disable=import-error
from .general import merge_two_dicts


np.seterr(divide='ignore', invalid='ignore')

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


        #TODO DELETE accurancy
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
        :param dict1: Dictionary 1 (dict)
        :param dict2: Dictionary 2 (dict)
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



        FP = (conf_matrix.sum(axis=0) - np.diag(conf_matrix))
        FN = (conf_matrix.sum(axis=1) - np.diag(conf_matrix))
        TP = (np.diag(conf_matrix))
        TN = (conf_matrix.sum() - (FP + FN + TP))

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        #Not Needed FPR = FP/(FP+TN)
        # False negative rate
        #Not Needed FNR = FN/(TP+FN)
        # False discovery rate
        #Not Needed FDR = FP/(TP+FP)
        # false omission rate
        #Not Needed FOR = FN/(TP+FP)

        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)

        # F1 Score
        F1 = (2*TP)/(2*TP+FP+FN)

        #TODO ???????? Temporary
        #Correction Terms
        TPR = np.where(np.isnan(TPR)==True, 0, TPR)
        TNR = np.where(np.isnan(TNR)==True, 0, TNR)
        PPV = np.where(np.isnan(PPV)==True, 0, PPV)
        NPV = np.where(np.isnan(NPV)==True, 0, NPV)
        #Not Needed FPR = np.where(np.isnan(FPR)==True, 0, FPR)
        #Not Needed FNR = np.where(np.isnan(FNR)==True, 0, FNR)
        #Not Needed FDR = np.where(np.isnan(FDR)==True, 0, FDR)
        #Not Needed FOR = np.where(np.isnan(FOR)==True, 0, FOR)
        ACC = np.where(np.isnan(ACC)==True, 0, ACC)
        F1  = np.where(np.isnan(F1)==True, 0, F1)

        # TPR = np.where(np.isposinf(TPR)==True, 1, TPR)
        # TNR = np.where(np.isposinf(TNR)==True, 1, TNR)
        # PPV = np.where(np.isposinf(PPV)==True, 1, PPV)
        # NPV = np.where(np.isposinf(NPV)==True, 1, NPV)
        # FPR = np.where(np.isposinf(FPR)==True, 1, FPR)
        # FNR = np.where(np.isposinf(FNR)==True, 1, FNR)
        # FDR = np.where(np.isposinf(FDR)==True, 1, FDR)
        # FOR = np.where(np.isposinf(FOR)==True, 1, FOR)
        # ACC = np.where(np.isposinf(ACC)==True, 1, ACC)
        # F1  = np.where(np.isposinf(F1)==True, 1, F1)



        print(conf_matrix)

        print("\n")
        print("FP: ",               FP)
        print("FN: ",               FN)
        print("TP: ",               TP)
        print("TN: ",               TN)
        print("\n")




        print("\n Sensitivity: ",               TPR, TPR.mean(),
              "\n Specificity: ",               TNR, TNR.mean(),
              "\n positive predictive value: ", PPV, PPV.mean(),
              "\n Negative predictive value: ", NPV, NPV.mean(),
              #Not Needed "\n false positive rate:",        FPR, FPR.mean(),
              #Not Needed "\n False negative rate: ",       FNR, FNR.mean(),
              #Not Needed "\n False discovery rate: ",      FDR, FDR.mean(),
              #Not Needed "\n false omission rate: ",       FOR, FOR.mean(),
              "\n F1 Score: ",                  F1, F1.mean(),
              "\n Accurancy: ",                 ACC, ACC.mean())
        print("\n")

        ret_dict = { dataset+"_loss": self.averagemeter[dataset]["loss"].avg,
                     dataset+"_TPR": TPR.mean(),
                     dataset+"_TNR": TNR.mean(),
                     dataset+"_PPV": PPV.mean(),
                     dataset+"_NPV": NPV.mean(),
                     #Not Needed dataset+"_FPR": FPR.mean(),
                     #Not Needed dataset+"_FNR": FNR.mean(),
                     #Not Needed dataset+"_FDR": FDR.mean(),
                     #Not Needed dataset+"_FOR": FOR.mean(),
                     dataset+"_F1": F1.mean(),
                     dataset+"_ACC": ACC.mean()}
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

        #Saving as CSV for Each Epoch (Averaged Values)
        filename = os.path.join(self.path, func + ".csv")
        file_exists = os.path.isfile(filename)
        with open(filename, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            if not file_exists:
                writer.writeheader()
            writer.writerow(to_be_saved_dict)



        #Reset conf_matrix_epoch
        for i in self.conf_matrix_epoch:
            self.conf_matrix_epoch[i] = deepcopy(self.conf_matrix_template)

        #Reset all AverageMeter
        for i in self.averagemeter:
            for j in self.averagemeter[i]:
                self.averagemeter[i][j] = AverageMeter()


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
        #TODO DELETE accurancy
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
