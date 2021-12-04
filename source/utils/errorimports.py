


def read_heif(*inp):
    """
    Handling Error from using pyheif.read_heif on Windows!
    """
    assert None, f"Error Processing Input: {inp}!\n \
                 pyheif does not Exist on Windows.\n \
                 Switch to Linux or Mac to Process HEIC Pictures! \n \
                 Documentation-URL: https://pypi.org/project/pyheif"
