



from mapproxy.util.ext.dictspec.validator import validate, ValidationError
from mapproxy.util.ext.dictspec.spec import one_of, number, required, combined


#Infos https://pytorch.org/docs/stable/optim.html
#https://github.com/mapproxy/mapproxy/blob/master/mapproxy/util/ext/dictspec/spec.py

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

hyperparameter = { 'RandomHorizontalFlip': number(),
                   'RandomRotation_Degree': number(),
                   'Normalize':{
                        'mean': [number()],
                        'std' : [number()],
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
    except ValidationError as ex:
        return ex.errors, ex.informal_only
    else:
        return [], True
