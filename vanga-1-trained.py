from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# exel - 2 == slice

#btc = 2675

class vanga:
    def __init__(self, btc):
        self.btc = btc

        razmer_stupeni = 24
        k = 0.1

        f = open('C:/MLProjects/venv7/vanga-1/btc/BTC-s2s.txt', 'w')

        a = pd.read_csv('C:/MLProjects/venv7/vanga-1/btc/BTC-USD.csv', sep=';', usecols=[1, 4])

        list_chars = []

        a['mid'] = (a['Open'] + a['Close']) / 2
        var = ((a['Open'] - a['Close']) / a['mid'])
        max_var = (max(var))*k
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
        char_indices = {i: c for i, c in enumerate(sorted(np.linspace(-max_var, max_var, len(alphabet))))}

        def int_to_char(int_prev, int_next, int_mid):
            sentence = []
            var1 = (int_next - int_prev) / int_mid
            for i in range(0, len(alphabet) - 1):
                if char_indices[i] < var1 <= char_indices[i + 1]:
                    sentence.append(alphabet[i])

            sentence2 = (''.join(map(str, sentence)))
            return sentence2

        price_btc0 = []
        price_btc1 = []

        sentence3 = []
        for stupenb in range(btc, razmer_stupeni + btc + 3):
            chars = a.at[stupenb, 'Open']
            chars2 = (a.at[stupenb, 'Close'])
            chars3 = (a.at[stupenb, 'mid'])
            price_btc0.append(chars3)
            chars3_2 = (a.at[stupenb + razmer_stupeni - 1, 'mid'])
            price_btc1.append(chars3_2)
            sentence3.append(int_to_char(chars, chars2, chars3))
        list_chars.append(''.join(map(str, sentence3)))

        price_btc_to_itog = (';'.join(map(str, price_btc1)))

        listik = []
        for i in range(5, len(sentence3) + 1):
            listik.append(''.join(map(str, sentence3[-i:])))

        list_output = ['qwertyuiopasdfghjklzxcvbnm\t','qwertyuiopasdfghjklzxcvbnm\n']

        for i in range(0, 18):
            c = listik[i] + '\tqwertyuiopasdfghjklzxcvbnm\n'
            list_output.append(c)

        output_chars2 = (''.join(map(str, list_output)))

        f.write(output_chars2)
        f.close()

        batch_size = 64  # Batch size for training.
        epochs = 10  # Number of epochs to train for.
        latent_dim = 256  # Latent dimensionality of the encoding space.
        num_samples = 10000  # Number of samples to train on.
        # Path to the data txt file on disk.
        data_path = 'BTC-s2s.txt'

        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        #print('Number of samples:', len(input_texts))
        #print(input_text)
        #print(target_text)
        #print('Number of unique input tokens:', num_encoder_tokens)
        #print('Number of unique output tokens:', num_decoder_tokens)
        #print('Max sequence length for inputs:', max_encoder_seq_length)
        #print('Max sequence length for outputs:', max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.load_weights('BTC-5-25-5_weights.h5')
        #load_model('s2s.h5')

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, target_token_index['\t']] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = reverse_target_char_index[sampled_token_index]
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '\n' or
                        len(decoded_sentence) > max_decoder_seq_length):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1.

                # Update states
                states_value = [h, c]

            return decoded_sentence

        f = open('BTC_outputs2s.txt', 'w')

        for seq_index in range(19):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(input_seq)
            f.write(decoded_sentence)

        f = open('BTC_itog.csv', 'w')

        a = pd.read_csv('C:/MLProjects/venv7/vanga-1/btc/BTC-USD.csv', sep=';', usecols=[1, 4])

        a['mid'] = (a['Open'] + a['Close']) / 2
        var = ((a['Open'] - a['Close']) / a['mid'])
        max_var = (max(var))*0.1
        int_indices = {alphabet[i]: c for i, c in enumerate(sorted(np.linspace(-max_var, max_var, len(alphabet))))}

        sentence3 = []
        for i in range(0, razmer_stupeni + 2):
            sentence3.append(str(i) + ';')
        sentence3.append('\n')

        outputs2s = open('BTC_outputs2s.txt', 'r')

        for word in outputs2s:
            for place in range(0, 2):
                if len(word) == 1:
                    sentence3.append('\n')
                else:
                    if place == 0:
                        begin = int_indices[word[0]] * a.at[btc + razmer_stupeni - 1, 'mid'] + (a.at[btc + razmer_stupeni - 1, 'mid'])
                        sentence3.append(begin)
                        sentence3.append(';')
                    elif place == 1:
                        for i in range(1, len(word)):
                            if word[i] != '\n':
                                sentence3.append(int_indices[word[i]] * begin + begin)
                                sentence3.append(';')
                                begin = int_indices[word[i]] * begin + begin
                            else:
                                sentence3.append('\n')

        sentence2 = (''.join(map(str, sentence3)))

        f.write(sentence2)
        f.close()

        btc_pd = pd.DataFrame(price_btc1)
        itogo = pd.read_csv('BTC_itog.csv', sep=';')
        itogoT = itogo.T.drop([0], axis=1)
        itogoTM = itogoT.mean(axis=1)
        itogoTSTD = itogoT.std(axis=1)
        itogo_all = pd.DataFrame([itogoTM, itogoTM + itogoTSTD, itogoTM - itogoTSTD])
        plt.plot(btc_pd)
        plt.plot(itogo_all.T, linestyle=":")
        plt.savefig('{}.png'.format(btc))
        plt.clf()

for i in range(3000, 7000):
    vanga(i)