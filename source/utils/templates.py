#TODO Docstring
"""
TODO
"""

from torchvision.models import resnet18

allowed_fn = ["symmetry", "eye", "mouth", "forehead"]


# TODO lookup and grading and Model
house_brackmann_lookup = {
    "symmetry": {
        "enum":{
            "normal": 0,
            "min_asymm":1,
            "asymm":2},
        "model": resnet18(pretrained=True)
    },
    "eye": {
        "enum":{
            "complete":0,
            "incomplete":1},
        "model": resnet18(pretrained=True)
    },
    "mouth": {
        "enum":{
            "normal":0,
            "min_asymm":1,
            "asymm":2},
        "model":resnet18(pretrained=True)
    },
    "forehead": {
        "enum":{
            "normal":0,
            "min_asymm":1,
            "asymm":2},
        "model": resnet18(pretrained=True)
    }
}


house_brackmann_template = {
    "symmetry": None,
    "eye": None,
    "mouth": None,
    "forehead": None,
}

house_brackmann_grading = {
    #       symmetry,  eye,         forehead,    mouth
    "I"    : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"normal"   ,   "mouth":"normal"   },
    "II"   : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"min_asymm",   "mouth":"min_asymm"},
    "III"  : {"symmetry":"normal",   "eye":"complete"  ,   "forehead":"min_asymm",   "mouth":"min_asymm"},
    "IV"   : {"symmetry":"normal",   "eye":"incomplete",   "forehead":"min_asymm",   "mouth":"min_asymm"},
    "V"    : {"symmetry":"asymm" ,   "eye":"incomplete",   "forehead":"min_asymm",   "mouth":"asymm"    },
    "VI"   : {"symmetry":"asymm" ,   "eye":"incomplete",   "forehead":"asymm"    ,   "mouth":"asymm"    },
}
