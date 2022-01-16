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
"""

from hbmedicalprocessing.utils.config import LOGGER


# define the function blocks
def one(symmetry, eye, mouth, forehead): #pylint: disable=unused-argument
    """
    State One:

    :param symmetry:  label from symmetry as Value (int)
    :param eye:  label from eye as Value (int)
    :param mouth:  label from mouth as Value (int)
    :param forehead:  label from forehead as Value (int)
    :returns  True if something changes else False and number for next state
    """
    if mouth == 3:
        return 6, True
    if symmetry >= 1:
        return 5, True
    if eye == 1:
        return 4, True
    if forehead >= 1:
        return 3, True
    if mouth >= 1:
        return 2, True
    return 1, False

def two(symmetry, eye, mouth, forehead): #pylint: disable=unused-argument
    """
    State Two:

    :param symmetry:  label from symmetry as Value (int)
    :param eye:  label from eye as Value (int)
    :param mouth:  label from mouth as Value (int)
    :param forehead:  label from forehead as Value (int)
    :returns  True if something changes else False and number for next state
    """
    if mouth == 2:
        return 4, True
    return 2, False

def three(symmetry, eye, mouth, forehead): #pylint: disable=unused-argument
    """
    State Three:

    :param symmetry:  label from symmetry as Value (int)
    :param eye:  label from eye as Value (int)
    :param mouth:  label from mouth as Value (int)
    :param forehead:  label from forehead as Value (int)
    :returns  True if something changes else False and number for next state
    """
    if forehead == 2 or mouth == 2:
        return 4, True
    return 3, False

def four(symmetry, eye, mouth, forehead): #pylint: disable=unused-argument
    """
    State Four:

    :param symmetry:  label from symmetry as Value (int)
    :param eye:  label from eye as Value (int)
    :param mouth:  label from mouth as Value (int)
    :param forehead:  label from forehead as Value (int)
    :returns  True if something changes else False and number for next state
    """
    return 4, False

def five(symmetry, eye, mouth, forehead): #pylint: disable=unused-argument
    """
    State Five:

    :param symmetry:  label from symmetry as Value (int)
    :param eye:  label from eye as Value (int)
    :param mouth:  label from mouth as Value (int)
    :param forehead:  label from forehead as Value (int)
    :returns  True if something changes else False and number for next state
    """
    if symmetry == 2:
        return 6, True
    return 5, False

def six(symmetry, eye, mouth, forehead): #pylint: disable=unused-argument
    """
    State Six:

    :param symmetry:  label from symmetry as Value (int)
    :param eye:  label from eye as Value (int)
    :param mouth:  label from mouth as Value (int)
    :param forehead:  label from forehead as Value (int)
    :returns  True if something changes else False and number for next state
    """
    return 6, False

# map the inputs to the function blocks
options = {1 : one,
           2 : two,
           3 : three,
           4 : four,
           5 : five,
           6 : six,}



def hb_automata(symmetry, eye, mouth, forehead):
    """
    Automata

    Runs trough automata until nothing changes

    :param symmetry:  label from symmetry as Value (int)
    :param eye:  label from eye as Value (int)
    :param mouth:  label from mouth as Value (int)
    :param forehead:  label from forehead as Value (int)

    :returns  State (int)
    """
    LOGGER.debug("Automata: inital values - symm=%s, eye=%s, mouth=%s, forehead=%s", symmetry, eye, mouth, forehead)
    changed = True
    state = 1 #Starting Node

    while changed:
        LOGGER.debug("Automata: Iterate step - state=%s, changed=%s", state, changed)
        state, changed = options[state](symmetry, eye, mouth, forehead)

    return state
