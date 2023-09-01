"""
scripts/lib/read_yaml.py

Function to get the content of a yaml file.
"""

import yaml
import pathlib

def read_yaml(path: pathlib.Path) -> dict:
    """
    Function to get content of a yaml file.

    Args:
        path (pathlib.Path): Path to the yaml file

    Returns:
        d_output (dict): Content of the yaml file stored as a dictionary.
    """

    # Open file
    file = open(path)

    # Load the content to a dictionary
    d_output = yaml.load(file, Loader=yaml.FullLoader)

    # Close the file to avoid connection issues
    file.close()

    return d_output
