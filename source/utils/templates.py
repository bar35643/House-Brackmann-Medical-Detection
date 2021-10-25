#TODO Docstring
"""
TODO
"""

from torchvision.models import resnet18

allowed_fn = ["symmetry", "eye", "mouth", "forehead"]


# TODO lookup and grading and Model
house_brackmann_lookup = {
    "symmetry": {
        "lookup": ["normal", "min_asymm", "asymm"],
        "model": resnet18(pretrained=True)
    },
    "eye": {
        "lookup": ["complete", "incomplete"],
        "model": resnet18(pretrained=True)
    },
    "mouth": {
        "lookup": ["normal", "min_asymm", "asymm"],
        "model":resnet18(pretrained=True)
    },
    "forehead": {
        "lookup": ["normal", "min_asymm", "asymm"],
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
    "I"  : ["normal", "complete"  , "normal"   , "normal"   ],
    "II" : ["normal", "complete"  , "normal"   , "min_asymm"],
    "III": ["normal", "complete"  , "min_asymm", "min_asymm"],
    "IV" : ["normal", "incomplete", "min_asymm", "min_asymm"],
    "V"  : ["asymm" , "incomplete", "min_asymm", "asymm"    ],
    "VI" : ["asymm" , "incomplete", "asymm"    , "asymm"    ],
}
