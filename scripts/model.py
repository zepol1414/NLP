"""
scripts/model.py

Function to train and evaluate the model.
"""

################################################################################
# IMPORT MODULES
################################################################################
# Import libraries
from pathlib import Path
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Import user defined functions
from scripts.lib.read_yaml import read_yaml
from scripts.lib.print_metrics import print_metrics

################################################################################
# SETTINGS
################################################################################
# Read configuration file
d_config = read_yaml("config/config.yml")

# Read configuration options
str_dataset = d_config["config"]["dataset"]
str_dataset_file = d_config["datasets"][str_dataset]["file"]
bool_embedding = d_config["config"]["embedding"]
str_vectorizer = d_config["config"]["vectorizer"]
str_model = d_config["config"]["model"]["algorithm"]
test_size = d_config["config"]["model"]["test_size"]

# Initialise paths to directories
dir_out_model = Path(d_config["paths"]["output"]["model"])

################################################################################
# MAIN
################################################################################
# Define main function
def function_model(df):
    """
    Train and evaluate model.

    Args:
        df (pd.DataFrame): Dataset with two columns: text and category.

    Returns:
        None.
    """

    ############################################################################
    # CREATE FEATURE AND LABEL SETS
    ############################################################################
    if bool_embedding:
        X = df["text_emb"].values.tolist()
        y = df["category"]
    
    else:
        X = df["text"]
        y = df["category"]

    ############################################################################
    # SPLIT TRAIN TEST DATA
    ############################################################################
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=123
    )

    ############################################################################
    # CONVERT TEXT INTO NUMERICAL FEATURE
    ############################################################################
    if not bool_embedding:
        if str_vectorizer == "count":
           # Use the Bag of Words model (CountVectorizer) to convert the cleaned
           # text into numeric features. This is needed for training the machine
           # learning model
           vectorizer = CountVectorizer()
        elif str_vectorizer == "tfidf":
            # We can also do TfidfVectorizer, which is equivalent to CountVectorizer
            # followed by TfidfTransformer
            vectorizer = TfidfVectorizer()

        else:
            raise Exception(
                "Vectorizer {} is not implemented.".format(str_vectorizer)
            )
        
        # Transform X_train using the vectorizer
        X_train_vect = vectorizer.fit_transform(X_train)

        # Transform X_test using the vectorizer
        X_test_vect = vectorizer.transform(X_test)

    ############################################################################
    # TRAIN AND EVALUATE MODEL
    ############################################################################
    # Define model
    if str_model == "naive_bayes":
        model = MultinomialNB()

    elif str_model == "stochastic_gradient_descent":
        model = SGDClassifier(
            max_iter=1000, tol=1e-3
        )

    elif str_model == "gradient_boosting":
        model = GradientBoostingClassifier( 
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        )

    elif str_model == "random_forest":
        model = RandomForestClassifier(
            n_estimators=10, random_state=42, n_jobs=-1
        )

    else:
        raise Exception("Model {} is not implemented.".format(str_model))
    
    # Train and evaluate model
    if bool_embedding:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Define test dataset
        df_test = pd.concat({"text": df["text"], "category": y_test}, axis = 1)
        df_test = df_test[df_test["category"].notna()]

    else:
        model.fit(X_train_vect, y_train)
        y_pred = model.predict(X_test_vect)
        # Define test dataset
        df_test = pd.concat({"text": X_test, "category": y_test}, axis = 1)

    df_test = df_test.reset_index(drop=True)

    # Add predictions to test dataset
    df_test["category_pred"] = y_pred

    # Add boolean column
    df_test["OK"] = np.where(
        df_test["category"] == df_test["category_pred"], 1, 0
    )

    # Save results
    str_file_pred_test = "predictions_test_" + str_dataset_file
    path_file_pred_test = dir_out_model.joinpath(str_file_pred_test)
    df_test.to_csv(path_file_pred_test, index=False)

    # Save model
    str_file_model = "model_" + str_model + "_" + str_dataset + ".sav"
    path_file_model = dir_out_model.joinpath(str_file_model)
    pkl.dump(model, open(path_file_model, "wb"))

    ############################################################################
    # METRICS
    ############################################################################
    str_output_file_cm = str(dir_out_model.joinpath(f"cm_{str_model}_{str_dataset}.png"))
    print_metrics(y_test, y_pred, str_output_file_cm)

    return
