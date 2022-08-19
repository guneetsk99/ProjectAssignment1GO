"""
Streamlit application code to demonstrate the working of text classification
"""
import pickle

import streamlit as st
import numpy as np

import joblib
from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences

json_file = open('/home/guneet.k/PycharmProjects/PythonAssignment1/PythonProject/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/home/guneet.k/PycharmProjects/PythonAssignment1/PythonProject/model.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
with open('/home/guneet.k/PycharmProjects/PythonAssignment1/PythonProject/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# evaluate loaded model on test data

pipe_lr = joblib.load(open("/home/guneet.k/PycharmProjects/PythonAssignment1/PythonProject/emotion_classifier_pipe_lr", "rb"))
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}


def predict_lstm(docx):
    new_complaint = [docx]
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=100)
    pred = loaded_model.predict(padded)
    labels = ['neutral', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust', 'shame']
    return pred, labels[np.argmax(pred)]


def predict_emotions(docx):
    """
    Emotion Prediction using ML Model
    :param docx: Input sentence
    :return: Predicted Emotion
    """
    results = pipe_lr.predict([docx])
    return results[0]


def get_pred_prob(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    """
    Streamlit application design function for text classification
    """
    st.title("Emotion Classifier")
    menu = ["Test ML", "Test DL", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == 'Test ML':
        st.subheader("Emotion in Text using ML")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type your sentence")
            submit_text = st.form_submit_button(label='Press to detect emotion')
        if submit_text:
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_pred_prob(raw_text)

            with col1:
                st.success('Input Text')
                st.write(raw_text)
                st.success('Prediction')
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
            with col2:
                st.success("Prediction Probability")
                st.write(probability)

    elif choice == 'Test DL':
        st.subheader("Emotion in Text using LSTM")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type your sentence")
            submit_text = st.form_submit_button(label='Press to detect emotion')
        if submit_text:
            col1, col2 = st.columns(2)
            probability, prediction = predict_lstm(raw_text)

            with col1:
                st.success('Input Text')
                st.write(raw_text)
                st.success('Prediction')
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
            with col2:
                st.success("Prediction Probability")
                st.write(probability)
    else:
        st.subheader("About")


if __name__ == '__main__':
    main()
