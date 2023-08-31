"""
scripts/embedding.py

Function to embed the text column of the dataset.
"""

################################################################################
# IMPORT MODULES

# Import libraries
import sys
import spacy

sys.dont_write_bytecode = True  # Disable cache

################################################################################
# SETTINGS
################################################################################
# Load the spacy model
nlp = spacy.load("en_core_web_lg")

################################################################################
# MAIN
################################################################################
# Define main function
def function_embedding(df):
    """
    Embed the text column of the dataset.

    Args:
        df (pd.DataFrame): Dataset with two columns, text and category.

    Returns:
        pd.DataFrame.
    """

    # Create sentence and label lists
    sentences = df.text.values
    text_emb = [nlp(sentence) for sentence in sentences]
    embedding = [text.vector for text in text_emb]

    df["text_emb"] = embedding

    return df
    