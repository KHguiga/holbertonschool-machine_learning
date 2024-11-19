#!/usr/bin/env python3
"""
From Dictionary
"""
import pandas as pd


def from_dictionary():
    """
        function that creates a pd.DataFrame from a dictionary

    :return: DataFrame,
        The first column should be labeled First and have the
         0.0, 0.5, 1.0, and 1.5
        The second column should be labeled Second and have
         the values one, two, three, four
        The rows should be labeled A, B, C, and D, respectively
    """
    dataframe = pd.DataFrame(
        {'First': [0.0, 0.5, 1.0, 1.5],
         'Second': ["one", "two", "three", "four"],
         },
        index=list("ABCD")
    )
    return dataframe


df = from_dictionary()
