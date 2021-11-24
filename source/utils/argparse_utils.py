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


class SmartFormatter(argparse.HelpFormatter):
    """
    Source:
    https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
    """
    def _split_lines(self, text, width):
        """
        Formatting the Argparser using "R|" at the beginning.
        Prints like shown in help with the newlines
        example use:

            parser = ArgumentParser(description='test', formatter_class=SmartFormatter)
            parser.add_argument('-g', choices=['a', 'b', 'g', 'd', 'e'], default='a',
                                help="R|Some option, where\n"
                                     " a = alpha\n"
                                     " b = beta\n"
                                     " g = gamma\n"
                                     " d = delta\n"
                                     " e = epsilon")

            parser.parse_args()
        """
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)
