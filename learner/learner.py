import pyspectre
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
import tensorflow as tf
from sklearn.externals import joblib
import time
import sys
import util


def train():
    trainer = pyspectre.getTrainerStr()
    scaler = preprocessing.StandardScaler()
    num_classes = len(trainer)
    num_samples = 500000
    X = np.zeros((num_samples, 256))
    y = np.zeros((num_samples, num_classes))

    print("Sampling {} spectre.c training data...".format(num_samples))
    for i in range(num_samples):
        char_pos = i % num_classes
        sample = np.array(pyspectre.readMemoryByte(char_pos, True))
        if i % 100000 == 0:
            print(sample)
        X[i, :] = sample
        # create one hot encoding of known chars
        y[i] = np.eye(num_classes)[char_pos]

    # Standardize cache timings (zero mean and unit variance) to stabilize learning algorithm.
    print("Standardizing cache timings...")
    X = scaler.fit_transform(X)

    # Split data into random 75% and 25% subsets for training and testing
    print("Splitting into training and test data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation='relu', input_shape=(256,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    model.save('model.h5')
    joblib.dump(scaler, 'scaler.pkl')


def test_model():
    model = tf.keras.models.load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    trainer = pyspectre.getTrainerStr()
    secret = pyspectre.getSecretStr()

    start_time = time.time()
    guessed_secrets = []
    for _ in range(3):
        print("Sampling...")
        X = np.zeros((len(secret), 256))
        for i in range(len(secret)):
            X[i] = np.array(pyspectre.readMemoryByte(i, False))
        X = scaler.transform(X)
        guessed_chars = np.argmax(model.predict(X), axis=1)
        guessed_secret = "".join([trainer[x] for x in guessed_chars])
        guessed_secrets.append(guessed_secret)
    
    majority_guess = util.majority_vote(guessed_secrets)
    print("Accuracy:", util.get_accuracy(majority_guess, secret))
    print("Total time:", time.time()-start_time)

def inspect_timings():
    trainer = pyspectre.getTrainerStr()
    num_classes = len(trainer)
    scaler = joblib.load('scaler.pkl')

    for i in range(num_classes):
        sample = np.array(pyspectre.readMemoryByte(i, True))
        print(sample)
    print(np.argmin(sample))
    sample = scaler.transform(sample.reshape(1, -1))
    print(np.argmin(sample))


if __name__ == "__main__":
    args = ['train', 'inspect']
    if len(sys.argv) == 1:
        print("Testing model...")
        test_model()
    elif sys.argv[1] == args[0]:
        print("Training model...")
        train()
    elif sys.argv[1] == args[1]:
        print("Inspecting timings")
        inspect_timings()
    else:
        print("Unknown argument {}, known are {}".format(sys.argv[1], args))

    
