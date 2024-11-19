#!/usr/bin/env python3

import pandas as pd

def from_file(filename, delimiter):
    # Load the file into a pandas DataFrame using the specified delimiter
    return pd.read_csv(filename, delimiter=delimiter)
