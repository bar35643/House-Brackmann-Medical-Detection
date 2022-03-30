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
# - 2022-03-12 Final Version 1.0.0 (~Raphael Baumann)
"""

from pathlib import Path
import yaml

from mapproxy.util.ext.dictspec.validator import validate, ValidationError
from mapproxy.util.ext.dictspec.spec import one_of, number, required, combined


#Infos https://pytorch.org/docs/stable/optim.html
#https://github.com/mapproxy/mapproxy/blob/master/mapproxy/util/ext/dictspec/spec.py

#List of all Schedulers available on pytorch docs written down as Dictionarys with their
#allowed Values for crosschecking input config file


#lambdalr = {'LambdaLR':  {'lr_lambda': float(), 'last_epoch': int(), 'verbose': bool()}  }
#multiplicativelr = {'MultiplicativeLR':  {'lr_lambda': float(), 'last_epoch': int(), 'verbose': bool()}  }
steplr = {'StepLR':{required('step_size'): int(),
                    'gamma': number(),
                    'last_epoch': int(),
                    'verbose': bool()}  }
multisteplr = {'MultiStepLR':{required('milestones'): [number()],
                              'gamma': float(),
                              'last_epoch': int(),
                              'verbose': bool()}  }
constantlr = {'ConstantLR':{'factor': number(),
                            'total_iters': int(),
                            'last_epoch': int(),
                            'verbose': bool()}  }
linearlr = {'LinearLR':{'start_factor': number(),
                        'end_factor': number(),
                        'total_iters': int(),
                        'last_epoch': int(),
                        'verbose': bool()}  }
exponentiallr = {'ExponentialLR':{required('gamma'): number(),
                                  'last_epoch': int(),
                                  'verbose': bool()}  }
cosineannealinglr = {'CosineAnnealingLR':{required('T_max'): int(),
                                          'eta_min': number(),
                                          'last_epoch': int(),
                                          'verbose': bool()}  }
reducelronplateau = {'ReduceLROnPlateau':{'mode': str(),
                                          'factor': number(),
                                          'patience': int(),
                                          'threshold': number(),
                                          'threshold_mode': str(),
                                          'min_lr': number(),
                                          'eps': number(),
                                          'verbose': bool()}  }
cycliclr = {'CyclicLR':{required('base_lr'): number(),
                        required('max_lr'): number(),
                        'step_size_up': int(),
                        'step_size_down': one_of(int(), None),
                        'mode': str(),
                        'gamma' : number(),
                        'scale_fn': None,
                        'scale_mode': str(),
                        'cycle_momentum': bool(),
                        'base_momentum': number(),
                        'max_momentum': number(),
                        'last_epoch': int(),
                        'verbose': bool()}  }
onecyclelr = {'OneCycleLR':{required('max_lr'): number(),
                            'total_steps': one_of(int(), None),
                            'epochs': one_of(int(), None),
                            'steps_per_epoch': one_of(int(), None),
                            'pct_start' : number(),
                            'anneal_strategy': str(),
                            'cycle_momentum': bool(),
                            'base_momentum': float(),
                            'max_momentum': number(),
                            'div_factor': number(),
                            'final_div_factor': number(),
                            'three_phase':bool(),
                            'last_epoch': int(),
                            'verbose': bool()}  }
cosineannealingwarmrestarts = {'CosineAnnealingWarmRestarts':{required('T_0'): int(),
                                                              'T_mult': int(),
                                                              'eta_min': number(),
                                                              'last_epoch': int(),
                                                              'verbose': bool()}  }

#List of all Optimizers available on pytorch docs written down as Dictionarys with their
#allowed Values for crosschecking input config file
adadelta = {'Adadelta':{'lr': number(),
                        'rho': number(),
                        'eps': number(),
                        'weight_decay': number()}  }
adagrad = {'Adagrad':{'lr': number(),
                      'lr_decay': number(),
                      'weight_decay': number(),
                      'initial_accumulator_value': number(),
                      'eps': number()}  }
adam = {'Adam':{'lr': number(),
                'betas': [number()],
                'eps': number(),
                'weight_decay': number(),
                'amsgrad': bool()}  }
adamw = {'AdamW':{'lr': number(),
                  'betas': [number()],
                  'eps': number(),
                  'weight_decay': number(),
                  'amsgrad': bool()}  }
sparseadam = {'SparseAdam':{'lr': number(),
                            'betas': [number()],
                            'eps': number()}  }
adamax = {'Adamax':{'lr': number(),
                    'betas': [number()],
                    'eps': number(),
                    'weight_decay': number()}  }
asgd = {'ASGD':{'lr': number(),
                'lambd': number(),
                'alpha': number(),
                't0': number(),
                'weight_decay': number()}  }
lbfgs = {'LBFGS':{'lr': number(),
                  'max_iter': int(),
                  'max_eval': one_of(int(), None),
                  'tolerance_grad': number(),
                  'tolerance_change': number(),
                  'history_size': int(),
                  'line_search_fn': one_of(str(), None)}  }
nadam = {'NAdam':{'lr': number(),
                  'betas': [number()],
                  'eps': number(),
                  'weight_decay': number(),
                  'momentum_decay': number()}  }
radam = {'RAdam':{'lr': number(),
                  'betas': [number()],
                  'eps': number(),
                  'weight_decay': number()}  }
rmsprop = {'RMSprop':{'lr': number(),
                      'alpha': number(),
                      'eps': number(),
                      'weight_decay': number(),
                      'momentum': number(),
                      'centered': bool()}  }
rprop = {'Rprop':{'lr': number(),
                  'etas': [number()],
                  'step_sizes': [number()]}  }
sgd = {'SGD':{'lr': number(),
              'momentum': number(),
              'dampening': number(),
              'weight_decay': number(),
              'nesterov': bool()}  }




#List of used Augmentations  written down as Dictionarys with their
#allowed Values for crosschecking input config file
hyperparameter = {required('imgsz'):{
                        required('symmetry'): [number()],
                        required('eye') : [number()],
                        required('mouth') : [number()],
                        required('forehead') : [number()],},
                   required('RandomHorizontalFlip'): number(),
                   required('Normalize'):{
                        required('mean'): [number()],
                        required('std') : [number()],
                   },
                   required('ColorJitter'):{
                        required('brightness'): float(),
                        required('contrast')  : float(),
                        required('saturation'): float(),
                        required('hue')       : float(),
                   },
                   required('RandomAffine'):{
                        required('degrees')  : number(),
                        required('translate'): [number()],
                   },
                   required('GaussianBlur'):{
                        required('kernel_size'): [number()],
                        required('sigma')      : [number()],
                   },
                   required('RandomAdjustSharpness'):{
                        required('probability'):float(),
                        required('val')        : number(),
                   },
}






def validate_yaml_config(inp):
    """
    This function validates the input to a spec

    :param inp: Input path to Yaml (path)
    :return array with errors, true/false (arr, bool)
    """
    yaml_spec = {
        required("optimizer"): combined(adadelta, adagrad, adam, adamw, sparseadam, adamax, asgd, lbfgs, nadam, radam, rmsprop, rprop, sgd),
        required("sequential_scheduler"): bool(),
        required("scheduler"): [combined(steplr, multisteplr, constantlr, linearlr, exponentiallr,
                                         cosineannealinglr, reducelronplateau, cycliclr, cosineannealingwarmrestarts)],
        required("hyp"):hyperparameter,
    }

    try:

        validate(yaml_spec, inp)
        if len(inp["optimizer"]) != 1:
            raise ValidationError("", [f'key optimizer has more than one element {list(inp["optimizer"].keys())}. Only one is allowed!'])
        return [], True
    except ValidationError as ex:
        return ex.errors, ex.informal_only


def validate_file(hyp:str):
    """
    This function validates the input to a spec from a file

    :param hyp: path to File (str)
    :return loaded YAML config as Dictionary (Dict)
    """
    pth = Path(hyp)
    assert hyp.endswith('.yaml') and pth.exists(), f"Error Path {hyp} has the wron ending or do not exist"
    with open(pth, 'r', encoding="UTF-8") as yaml_file:
        yml_hyp = yaml.safe_load(yaml_file)
        error, tru_fal = validate_yaml_config(yml_hyp)
        assert tru_fal, f"Error in YAML-Configuration (Path = {pth}): \n" + "\n".join(error)
    return yml_hyp
