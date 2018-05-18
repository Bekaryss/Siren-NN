from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np

class Neuron:

    def __init__(self, dense, dropout, lstm, activation, loss, optimizer, epochs, batch_size):
        self.dense = dense
        self.dropout = dropout
        self.lstm = lstm
        self.actication = activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.bach_size = batch_size

    def create_model(self, network_input, wights):
        model = Sequential()
        model.add(Dense(self.dense, input_shape=(network_input.shape[1], network_input.shape[2])))
        model.add(LSTM(self.lstm, return_sequences=False))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.dense))
        model.add(Dropout(self.dropout))
        model.add(Dense(network_input.shape[2]))
        model.add(Activation(self.actication))
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        if(wights != ''):
            model.load_weights(wights)

        return model

    def train_model(self, model, network_input, network_output):
        filepath = "weights_files/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        callbacks_list = []

        callbacks_list.append(ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'))

        callbacks_list.append(ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=10,
            verbose=0,
            mode='auto',
            min_delta=0.00001,
            cooldown=0,
            min_lr=0))

        model.fit(network_input, network_output, epochs=self.epochs, batch_size=self.bach_size, callbacks=callbacks_list, verbose=1)

    def predict(self, predModel, normalize_input, predict_lenght):
        pattern = np.zeros((1, normalize_input.shape[1], normalize_input.shape[2]))
        prediction_input = np.zeros((1, normalize_input.shape[1], normalize_input.shape[2]))

        pattern[0] = normalize_input[0]

        prediction_output = np.zeros((predict_lenght, normalize_input.shape[2]))
        print(pattern.shape[0], pattern.shape[1], pattern.shape[2])


        print(pattern[0, 0:99, 30:40])
        print("end")
        # generate predict_lenght notes
        for note_index in range(predict_lenght):
            prediction_input[0] = pattern[0]

            prediction = predModel.predict(prediction_input, verbose=0)
            for i in range(prediction.shape[0]):
                for j in range(prediction.shape[1]):
                    prediction[i][j] = round(prediction[i][j])

            print(prediction[0, 30:40])
            pattern[0, 0:pattern.shape[1] - 1, :] = pattern[0, 1:pattern.shape[1], :]
            pattern[0, pattern.shape[1] - 1, :] = prediction

            prediction_output[note_index, :] = prediction

        return prediction_output