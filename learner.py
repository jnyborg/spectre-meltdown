import pyspectre
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
import tensorflow as tf
from sklearn.externals import joblib


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
        sample = pyspectre.readMemoryByte(char_pos, True)
        X[i, :] = np.array(sample)
        # create one hot encoding of known chars
        y[i] = np.eye(num_classes)[char_pos]


    # Standardize cache timings (zero mean and unit variance) to stabilize learning algorithm.
    X = scaler.fit_transform(X)

    # Split data into random 75% and 25% subsets for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # clf = svm.LinearSVC(verbose=True)
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

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
    # score = model.evaluate(X_test, y_test, batch_size=32)
    # print("Result on test set: loss: {}, accuracy: {}".format(score[0], score[1]))
    model.save('model.h5')
    joblib.dump(scaler, 'scaler.pkl')


def test_model():
    model = tf.keras.models.load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    trainer = pyspectre.getTrainerStr()
    for _ in range(1):
        X = np.zeros((40, 256))
        for i in range(40):
            X[i] = np.array(pyspectre.readMemoryByte(i, False))
        X = scaler.transform(X)
        predictions = model.predict(X)
        chars = np.argmax(predictions, axis=1)
        print("".join([trainer[x] for x in chars]))

#train()
test_model()