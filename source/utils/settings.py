"""
TODO
Check internet connectivity
"""

from torchvision.models import resnet18

allowed_fn = ["symmetry", "eye", "mouth", "forehead"]


# TODO lookup and grading and Model
house_brackmann_lookup = {
    "symmetry": {
        "lookup": [],
        "model": resnet18(pretrained=True)
    },
    "eye": {
        "lookup": ["complete", "incomplete"],
        "model": resnet18(pretrained=True)
    },
    "mouth": {
        "lookup": [],
        "model":resnet18(pretrained=True)
    },
    "forehead": {
        "lookup": [],
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
    "I": [],
    "II": [],
    "III":[],
    "IV": [],

}
