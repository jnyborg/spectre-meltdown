import pyspectre
import pyspectre35
import pyspectre150
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

def get_pyspectre(keysize=None):
    if keysize is None or keysize is 1:
        return pyspectre
    elif keysize is 35:
        return pyspectre35
    elif keysize is 150:
        return pyspectre150
    else:
        print("Error, invalid get_pyspectre keysize argument")
        raise Exception()

def train():
    print("Training model...")
    trainer = pyspectre.getTrainerStr()
    scaler = preprocessing.StandardScaler()
    num_classes = len(trainer)
    num_samples = 500000
    X = np.zeros((num_samples, num_classes))
    y = np.zeros((num_samples, num_classes))

    print("Sampling {} spectre.c training data...".format(num_samples))
    for i in range(num_samples):
        char_pos = i % num_classes
        sample = np.take(np.array(pyspectre.readMemoryByte(char_pos, True)), [ord(x) for x in trainer])
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
        tf.keras.layers.Dense(200, activation='relu', input_shape=(num_classes,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_test, y_test))

    model.save('model.h5')
    joblib.dump(scaler, 'scaler.pkl')

def test_model(keys=1):

    model = tf.keras.models.load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    trainer = get_pyspectre(keys).getTrainerStr()
    secret = get_pyspectre(keys).getSecretStr()

    repetitions = 5
    samples = 3
    print("Testing model with keysize " + str(keys) + ". " + str(repetitions) + " repetitions of " + str(samples) + " samples each.")

    total_acc = 0
    total_runtime = 0
        
    for s in range(repetitions):
        start_time = time.time()
        guessed_secrets = []
        weights = np.zeros((len(secret), len(trainer)))
        for _ in range(samples):
            # print("Sampling...")
            X = np.zeros((len(secret), len(trainer)))
            for i in range(len(secret)):
                X[i] = np.take(np.array(get_pyspectre(keys).readMemoryByte(i, False)), [ord(x) for x in trainer])
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
    
    print(str(total_acc / repetitions * 100) + ",\t" + str(total_runtime / repetitions))

def inspect_timings():
    print("Inspecting timings")
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
    args = ['train', 'inspect', '1', '35', '150']
    if len(sys.argv) == 1 or sys.argv[1] == args[2]:
        test_model(1)
    elif sys.argv[1] == args[0]:
        train()
    elif sys.argv[1] == args[1]:
        inspect_timings()
    elif sys.argv[1] == args[3]:
        test_model(35)
    elif sys.argv[1] == args[4]:
        test_model(150)
    else:
        print("Unknown argument {}, known are {}".format(sys.argv[1], args))
