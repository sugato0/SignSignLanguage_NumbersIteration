import h5py
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json

try:

    with h5py.File("model.h5",'r') as f:
        print("fileAlreadyDownloaded")
        pass

except Exception as e:
    X = np.load("X.npz")
    y = np.load("y.npz")
    X = X["arr_0"]
    y = y["arr_0"]

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_trn)


    X_valid, X_train = X_trn[:100] / 255.0, X_trn[100:] / 255.0
    y_valid, y_train = y_trn[:100], y_trn[100:]

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[21,3]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))



    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

    model.evaluate(X_tst, y_tst)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# predictionss1 = loaded_model.predict(x1)
