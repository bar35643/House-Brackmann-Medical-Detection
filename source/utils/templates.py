#TODO Docstring
"""
TODO
"""
from torch import nn
from torchvision.models import resnet18


#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
allowed_fn = ["symmetry", "eye", "mouth", "forehead"]

model_2_label = resnet18(pretrained=True)
model_2_label.fc = nn.Linear(model_2_label.fc.in_features, 2)

model_3_label = resnet18(pretrained=True)
model_3_label.fc = nn.Linear(model_3_label.fc.in_features, 3)

model_4_label = resnet18(pretrained=True)
model_4_label.fc = nn.Linear(model_3_label.fc.in_features, 4)

house_brackmann_lookup = {
    "symmetry": {
        "enum":{
            "normal":0,
            "asymm" :1,
            "none"  :2},
        "model": model_3_label
    },
    "eye": {
        "enum":{
            "complete"  :0,
            "incomplete":1},
        "model": model_2_label
    },
    "mouth": {
        "enum":{
            "normal"   :0,
            "min_asymm":1,
            "asymm"    :2,
            "none"     :3},
        "model":model_4_label
    },
    "forehead": {
        "enum":{
            "normal"   :0,
            "min_asymm":1,
            "none"     :2},
        "model": model_3_label
    }
}


house_brackmann_template = {
    "symmetry": None,
    "eye": None,
    "mouth": None,
    "forehead": None,
}

image_input_template = {
    "1_rest": None,
    "2_lift_eyebrow": None,
    "3_smile_closed": None,
    "4_smile_open": None,
    "5_Duckface": None,
    "6_eye_closed_easy": None,
    "7_eye_closed_forced": None,
    "8_blow_nose": None,
    "9_depression_lower_lip": None,
}

house_brackmann_grading = {
    "I"    : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"normal"   ,   "mouth":"normal"   },
    "II"   : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"normal"   ,   "mouth":"min_asymm"},
    "III"  : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"min_asymm",   "mouth":"min_asymm"},
    "IV"   : {"symmetry":"normal",   "eye":"incomplete",   "forehead":"none"     ,   "mouth":"asymm"    },
    "V"    : {"symmetry":"asymm" ,   "eye":"incomplete",   "forehead":"none"     ,   "mouth":"asymm"    },
    "VI"   : {"symmetry":"none"  ,   "eye":"incomplete",   "forehead":"none"     ,   "mouth":"none"     },
}
