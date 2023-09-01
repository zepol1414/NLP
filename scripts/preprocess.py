"""
scripts/preprocess.py

Function to preprocess the text column of the dataset.
"""

################################################################################
# IMPORT MODULES
################################################################################
# Import libraries
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import user defined functions
from scripts.lib.read_yaml import read_yaml

################################################################################
# SETTINGS
################################################################################
# Read configuration file
d_config = read_yaml("config/config.yml")

# Read configuration options
bool_noun = d_config["config"]["pos"]["noun"]
bool_adv = d_config["config"]["pos"]["adv"]
bool_adj = d_config["config"]["pos"]["adj"]
bool_verb = d_config["config"]["pos"]["verb"]

# List nouns, adjectives, adverbs and verbs
ls_noun = ["NN", "NNS", "NNP", "NNPS"]
ls_adj = ["JJ", "JJS", "JJR"]
ls_adv = ["RB", "RBR", "RBS"]
ls_verb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

# Keep needed parts of speech (pos)
ls_tag = []
if bool_noun:
    ls_tag += ls_noun
if bool_adj:
    ls_tag += ls_adj
if bool_adv:
    ls_tag += ls_adv
if bool_verb:
    ls_tag += ls_verb

# Download everything from nltk library
nltk.download("all", quiet=True)

################################################################################
# MAIN
################################################################################
def function_preprocess(df):
    """
    Preprocess the text column of the dataset.

    Args:
        df (pd.DataFrame): Dataset with two columns, text and category.
    
    Returns:
        pd.DataFrame
    """

    # Create a list with texts
    text = df["text"].to_list()

    # Word lemmatizer
    lemmatizer = WordNetLemmatizer()
    corpus = []
    for i in range(len(text)):
        # Step 1: remove special characters
        r = re.sub("[^a-zA-Z]", " ", text[i])
        # Step 2: convert to lower case
        r = r.lower()
        # Step 3: split into words
        r = r.split()
        # Step 4: drop stopper words
        r = [word for word in r if word not in stopwords.words("english")]
        # Step 5: tagging and selecting parts of speech (POS)
        r_tagged = nltk.pos_tag(r)
        r = [t[0] for t in r_tagged if t[1] in ls_tag]
        # Step 6: lemmatize words
        r = [lemmatizer.lemmatize(word) for word in r]
        # Step 7: Bring back a sentence
        r = " ".join(r)
        corpus.append(r)

    # Assign corpus to data["text"]
    df["text"] = corpus

    return df






















