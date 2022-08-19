"""
The Following file helps in organising the Text Classification using LSTM
It contains preprocessing, training, evaluation and model saving
"""
import pickle

from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.models import Sequential
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from PythonProject.properties import application_properties

labels = {'neutral': 0,
          'joy': 1,
          'sadness': 2,
          'fear': 3,
          'anger': 4,
          'surprise': 5,
          'disgust': 6,
          'shame': 7,
          }


# def tweet_process(x):
#     z = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\\w+:\\\/\\\/\\S+)", " ", x).split())
#     return z


# saving


# loading

def lstm_preprocess(df):
    """
    Preprocessing Input Data for LSTM based Text Classification model training
    :param df: The input dataframe for training
    :return: The test and train split for the model training function
    """
    X_train = df['Clean_Text']
    # X_train = X_train.apply(tweet_process)

    tokenizer = Tokenizer(num_words=application_properties.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
    tokenizer.fit_on_texts(X_train.values)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    X = tokenizer.texts_to_sequences(X_train.values)
    X = pad_sequences(X, maxlen=application_properties.MAX_SEQUENCE_LENGTH)
    input_length = X.shape[1]
    df['Emotion'] = df['Emotion'].map(labels)
    Y = np_utils.to_categorical(df['Emotion'])

    print('Shape of label tensor:', Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=application_properties.test_size,
                                                        random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    return X_train, X_test, Y_train, Y_test, input_length


def lstm_train(X_train, X_test, Y_train, Y_test, input_length):
    """
    This function contains training of LSTM for Emotion based text classification
    """
    model = Sequential()
    model.add(
        Embedding(application_properties.MAX_NB_WORDS, application_properties.EMBEDDING_DIM, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss=application_properties.loss, optimizer=application_properties.optimizer, metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=application_properties.epochs,
                        batch_size=application_properties.batch_size,
                        callbacks=[EarlyStopping(patience=3, min_delta=0.0001)])
    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
