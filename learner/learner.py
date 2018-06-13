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

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save('model.h5')
    joblib.dump(scaler, 'scaler.pkl')


def test_model():
    model = tf.keras.models.load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    trainer = pyspectre.getTrainerStr()
    secret = pyspectre.getSecretStr()

    total_acc = 0
    total_runtime = 0
    total_runs = 3
    samples = 4
    
    for s in range(total_runs):
        start_time = time.time()
        guessed_secrets = []
        weights = np.zeros((len(secret), len(trainer)))
        for _ in range(samples):
            print("Sampling...")
            X = np.zeros((len(secret), 256))
            for i in range(len(secret)):
                X[i] = np.array(pyspectre.readMemoryByte(i, False))
            X = scaler.transform(X)
            prediction = model.predict(X)
            guessed_chars = np.argmax(prediction, axis=1)
            guessed_secret = "".join([trainer[x] for x in guessed_chars])
            weights += prediction
            guessed_secrets.append(guessed_secret)
        
        weight_to_char_map = {char: index for index, char in enumerate(trainer)}
        majority_guess = util.majority_vote(guessed_secrets, weights, weight_to_char_map)
        acc = util.get_accuracy(majority_guess, secret)
        total_acc += acc
        elapsed = time.time()-start_time
        total_runtime += elapsed
        print("Accuracy:", acc)
        print("Total time:", elapsed)

    print("Average accuracy:", (total_acc / total_runs))
    print("Average time:", (total_runtime / total_runs))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("Training model...")
        train()
    else:
        print("Testing model...")
        test_model()

    