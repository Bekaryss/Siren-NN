import glob
import numpy as np
from music21 import converter, instrument, note, chord, duration, stream, midi

class MidiUtils:
    midiPartsList = []

    def __init__(self, midi_file_path, minNote, maxNote, sequence_length, time_step, music_name, start_time, dur_time):
        self.midi_file_path = midi_file_path
        self.minNote = int(minNote)
        self.maxNote = int(maxNote)
        self.range = maxNote - minNote
        self.sequence_length = sequence_length
        self.time_step = time_step
        self.music_name = music_name
        self.start_time = start_time
        self.dur_time = dur_time


    def preprocessing(self):
        sqList = []
        inputList = []
        outputList = []
        self.get_midi_data()

        for item in self.midiPartsList:
            nd, matrix_length = self.get_notes(item)
            sq = self.get_matrix(nd, matrix_length)
            sqList.append(sq)
            # inputList.append(network_input)
            # outputList.append(network_output)

        # network_input_sequence = self.sequence_join(inputList)
        # network_output_sequence = self.sequence_join(outputList)
        sequence = self.sequence_join(sqList)
        network_input, network_output = self.IO_create(sequence)
        # print("Network input Sequence Created. Size: ", network_input_sequence.shape[0], network_input_sequence.shape[1], network_input_sequence.shape[2])
        # print("Network output Sequence Created. Size: ", network_output_sequence.shape[0], network_output_sequence.shape[1])
        print("Network input Sequence Created. Size: ", network_input.shape[0], network_input.shape[1])
        print("Network output Sequence Created. Size: ", network_output.shape[0], network_output.shape[1])

        return network_input, network_output

    def postprocessing_get_midi(self, path):
        sqList = []
        # inputList = []
        # outputList = []
        self.get_midi_file(path)

        for item in self.midiPartsList:
            print(item)
            nd, matrix_length = self.get_notes(item)
            sq = self.get_matrix(nd, matrix_length)
            sqList.append(sq)
            # inputList.append(network_input)
            # outputList.append(network_output)

        sequence = self.sequence_join(sqList)
        network_input, network_output = self.IO_create(sequence)
        # print("Network input Sequence Created. Size: ", network_input_sequence.shape[0], network_input_sequence.shape[1], network_input_sequence.shape[2])
        # print("Network output Sequence Created. Size: ", network_output_sequence.shape[0], network_output_sequence.shape[1])
        print("Network input Sequence Created. Size: ", network_input.shape[0], network_input.shape[1], network_input.shape[2])
        print("Network output Sequence Created. Size: ", network_output.shape[0], network_output.shape[1])

        # network_input_sequence = self.sequence_join(inputList)
        # network_output_sequence = self.sequence_join(outputList)
        # print("Network input Sequence Created. Size: ", network_input_sequence.shape[0],
        #       network_input_sequence.shape[1], network_input_sequence.shape[2])
        # print("Network output Sequence Created. Size: ", network_output_sequence.shape[0],
        #       network_output_sequence.shape[1])
        return network_input, network_output

    def postprocessing(self, matrix):
        prediction_output = self.get_predict_dictionary(matrix)
        print("Get Dictionary!")
        self.create_midi(prediction_output)

    def get_midi_data(self):
        self.midiPartsList.clear()
        count = 0
        for file in glob.glob(self.midi_file_path):
            print(file)
            midifile = converter.parse(file)
            self.midiPartsList.append(midifile)
            count += 1
        print('Got all midi file: ', count)


    def get_midi_file(self, path):
        self.midiPartsList.clear()
        count = 0
        for file in glob.glob(path):
            midifile = converter.parse(file)
            self.midiPartsList.append(midifile)
            count += 1
        print('Got all midi file: ', count)

    def get_notes(self, midi_file):
        notes = []
        chords = []
        notesDict = {n: [] for n in range(self.minNote, self.maxNote)}
        notes_to_parse = None

        try:  # file has instrument parts
            parts = instrument.partitionByInstrument(midi_file)
            notes_to_parse = parts.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi_file.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(element)
            elif isinstance(element, chord.Chord):
                notes.append(element)

        totalCount = len(notes) + len(chords)

        count = 0
        for element in notes:
            if isinstance(element, note.Note):
                if notesDict.__contains__(element.pitch.midi):
                    notesDict[element.pitch.midi] += [[count, element.offset, element.duration.quarterLength]]
            if isinstance(element, chord.Chord):
                for item in element:
                    if notesDict.__contains__(item.pitch.midi):
                        notesDict[item.pitch.midi] += [[count, element.offset, element.duration.quarterLength]]
            count = count + 1
        print("Get notes Dictionary!")
        print(notesDict)
        return notesDict, totalCount

    def get_matrix(self, notes_dict, time_total):
        sequence = np.zeros((time_total, len(notes_dict) + self.start_time + self.dur_time))

        print(sequence.shape[0], sequence.shape[1])

        for element in notes_dict:
            for (position, start, dur) in notes_dict[element]:
                start_t = (int)(start / self.time_step)
                dur_t = (int)(dur / self.time_step)
                start_b = np.binary_repr(start_t, width=self.start_time)
                dur_b = np.binary_repr(dur_t, width=self.dur_time)
                sequence[position, self.range : self.range + self.start_time] = np.array([int(i) for i in start_b], dtype=int)
                sequence[position, self.range + self.start_time : self.range + self.start_time+ self.dur_time] = np.array([int(i) for i in dur_b], dtype=int)
                sequence[position, element - self.minNote] = 1
        print("Create matrix")
        return sequence

    def IO_create(self, matrix):
        network_input = np.zeros((matrix.shape[0] - self.sequence_length, self.sequence_length, self.range + self.start_time+ self.dur_time))
        network_output = np.zeros((matrix.shape[0] - self.sequence_length, self.range + self.start_time + self.dur_time))
        for i in range(0, matrix.shape[0] - self.sequence_length, 1):
            network_input[i, :, :] = matrix[i:i + self.sequence_length, :]
            network_output[i, :] = matrix[i + self.sequence_length, :]
        return network_input, network_output

    def sequence_join(self, sequence_list):
        return np.concatenate(sequence_list, axis=0)


    def get_predict_dictionary(self, prediction_output):
        prediction_notes = {n + 30: [] for n in range(prediction_output.shape[1])}

        for tick in range(prediction_output.shape[0]):
            for item in range(prediction_output.shape[1]):
                if (item < self.range):
                    if (prediction_output[tick, item] == 1):
                        pre_start = np.array(prediction_output[tick, self.range : self.range + self.start_time], np.integer)
                        pre_dur = np.array(prediction_output[tick, self.range + self.start_time : self.range + self.start_time+ self.dur_time], np.integer)
                        d_start = int("".join(str(x) for x in pre_start), 2) * self.time_step
                        d_dur = int("".join(str(x) for x in pre_dur), 2) * self.time_step
                        prediction_notes[item + self.minNote] += [[d_start, d_dur]]
                        print(d_start, d_dur)
                else:
                    break
        print("Get dictionary notes!")
        return prediction_notes



    def create_midi(self, prediction_notes):
        all_notes = []

        for key, value in prediction_notes.items():
            for element in value:
                if (element[0] >= 0):
                    d = duration.Duration(element[1] - element[0])
                    new_note = note.Note(int(key))
                    new_note.offset = element[0]
                    new_note.duration = d
                    new_note.storedInstrument = instrument.Piano()
                    all_notes.append(new_note)

        midi_stream = stream.Stream(all_notes)
        midi_stream.write('midi', fp=self.music_name)
        print("Midi file successfully created!")
