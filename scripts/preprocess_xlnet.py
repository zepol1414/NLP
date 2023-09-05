"""
scripts/preprocess_xlnet.py

Function to preprocess the dataset as needed for XLNet model.
"""

################################################################################
# IMPORT MODULES
################################################################################
# Import libraries
import pandas as pd
from transformers import XLNetTokenizer
from keras_preprocessing.sequence import pad_sequences

################################################################################
# MAIN
################################################################################
def function_preprocess_xlnet(df):
    """
    Preprocess the dataset as needed for XLNet model.

    Args:
        df (pd.DataFrame): Dataset with two columns, text and category.
    
    Returns:
        pd.DataFrame
    """

    # One hot encoding
    df = pd.get_dummies(data=df, columns=["category"], dtype=int)
    df["one_hot_labels"] = df[[col for col in df if col.startswith("category_")]].values.tolist()

    # Create sentence list
    ls_sentences = df["text"].values.tolist()

    # We need to add special tokens at the end of each sentence for XLNet to work
    # properly
    ls_sentences = [sentence + " [SEP] [CLS]" for sentence in ls_sentences]

    # Use XLNet tokenizer to convert the sentences into tokens in XLNet's vocabulary
    tokenizer = XLNetTokenizer.from_pretrained(
        "xlnet-base-cased", do_lower_case=True
    )
    ls_tokenized_sentences = [
        tokenizer.tokenize(sentence) for sentence in ls_sentences
    ]

    # Use XLNet tokenizer to convert the tokens to their index numbers in XLNet's
    # vocabulary
    ls_input_ids = [
        tokenizer.convert_tokens_to_ids(tokenized_sentence)
        for tokenized_sentence in ls_tokenized_sentences
    ]

    # Pad input tokens
    ls_input_ids = pad_sequences(
        ls_input_ids, dtype="long", truncating="post", padding="post"
    ).tolist()

    # Create attention masks in order to avoid the model to pay attention to 
    # invalid tokens
    ls_attention_masks =[]
    for sequence in ls_input_ids:
        sequence_mask = [int(i > 0) for i in sequence]
        ls_attention_masks.append(sequence_mask)

    # Add input ids and attention masks to the dataframe
    df["features"] = ls_input_ids
    df["masks"] = ls_attention_masks

    return df









