"""
Main Function coordinates the training of ML Model
"""
import argparse
import os

import neattext.functions as nfx
import pandas as pd

from PythonProject.Training.ml_train import mlTrain
from PythonProject.Training.deep_learning_train import lstm_train, lstm_preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emotion Based Text Classification')
    parser.add_argument('--train_data_csv', type=str, required=False, help='Enter path of training data CSV')
    parser.add_argument('--training_type', type=int, required=False, help="Enter 0 for ML and 1 for DL")
    parser.add_argument('--run_demo', type=int, required=False, help="Enter 1 to run Demo or any other number for "
                                                                     "training")
    args = parser.parse_args()
    if args.run_demo == 1:
        os.system("streamlit run /home/guneet.k/PycharmProjects/PythonAssignment1/APP/app.py")
    else:
        df = pd.read_csv(args.train_data_csv)
        print("The Values Count of Emotions in Training Data", df['Emotion'].value_counts())
        df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
        df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
        if args.training_type == 1:
            X_train, X_test, Y_train, Y_test, input_length = lstm_preprocess(df)
            lstm_train(X_train, X_test, Y_train, Y_test, input_length)
        elif args.train_type == 0:
            mlTrain(df)
