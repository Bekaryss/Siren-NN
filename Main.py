from MidiUtils import MidiUtils
from Neuron import Neuron
from sys import stdin
import glob
import os

midiUtil = MidiUtils('midi_songs/*.mid', 30, 100, 100, 0.25, 'output.mid', 14, 12)
newron = Neuron(128, 0.1, 128, 'sigmoid', 'binary_crossentropy', "rmsprop", 500, 64)


def train():
    list_of_files = glob.glob('weights_files/*')
    latest_file = ''
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    input_network, output_network = midiUtil.preprocessing()
    print(input_network.shape[0], input_network.shape[1], input_network.shape[2])
    model = newron.create_model(input_network, latest_file)
    newron.train_model(model, input_network, output_network)

def predict():
    list_of_files = glob.glob('weights_files/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    input_network, output_network = midiUtil.postprocessing_get_midi('predict_song/*')
    model = newron.create_model(input_network, latest_file)
    preditin_matrix = newron.predict(model, input_network, 900)
    midiUtil.postprocessing(preditin_matrix)


if __name__ == '__main__':
    print("Enter 0 for training")
    x = stdin.read(1)
    if(int(x) == 0):
        train()
    else:
        predict()
