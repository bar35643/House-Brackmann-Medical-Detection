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

import argparse

def restricted_val_split(inp):
    """
    argparse multicasting input Value

    Alowwing only Variables with the Value: None, int or float in Intervall [0.0, 1.0]

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

    Fixing the issues of line length from the argparse
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
