"""
This file cover the training of Emotion Text Classification using ML technique
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from Training import application_properties


def mlTrain(df):
    """
    The following functions helps in training a text classification model using Machine Learning Techniques
    :param df: Input training data
    """
    X_features = df['Clean_Text']
    y_labels = df['Emotion']
    x_train, x_test, y_train, y_test = train_test_split(X_features, y_labels,
                                                        test_size=application_properties.test_size, random_state=42)
    pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])
    pipe_lr.fit(x_train, y_train)
    pipe_lr.score(x_test, y_test)
    import joblib
    pipeline_file = open("../emotion_classifier_pipe_lr", "wb")
    joblib.dump(pipe_lr, pipeline_file)
    pipeline_file.close()
