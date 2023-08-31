"""
scripts/lib/exploratory_analysis.py

Function to explore the dataset.
"""

import matplotlib.pyplot as plt

def exploratory_analysis(df, str_output_file_categories_barplot=None):
    """
    Exploratory analysis of the dataset

    Args:
        df (pd:Dataframe): Dataset with two columns: text and category.
        str_output_file_categories_barplot (str, optional): Name of the file relative to
                                                            the root folder of the
                                                            repository to save a plot with
                                                            the barplot of the categories
                                                            Defaults to None.
    Returns: None
    """

    # Number of rows
    print("Number of rows: ", len(df))

    # Statistics of the categories
    print("Categories barplot: ")
    fig = df["category"].value_counts(normalize=True).plot.bar().get_figure()
    if str_output_file_categories_barplot is not None:
        print(f"Saving file with barplot of the categories to {str_output_file_categories_barplot}")
        fig.savefig(str_output_file_categories_barplot, bbox_inches="tight")

    return