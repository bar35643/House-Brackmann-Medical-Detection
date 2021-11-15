import argparse

def restricted_val_split(inp):
    """
    argparse multicasting input Value

    :param inp: input data (str)
    :returns Formatted inp (int, float, NoneType)
    """

    try:
        if inp == "None":
            inp = None
        elif inp.isdigit():
            inp = int(inp)
        else:
            inp = float(inp)
            if inp < 0.0 or inp > 1.0:
                raise argparse.ArgumentTypeError(f"Value {inp} not in range [0.0, 1.0]")
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"Value {inp} not a float in range [0.0, 1.0], int or None") from err
    return inp
