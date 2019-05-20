from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, TimeDistributed, Bidirectional, dot, add
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

class Seq2SeqModel:
    def __init__(self, encoder_vocab_size, decoder_vocab_size, enc_seq_len, dec_seq_len):
        self.comvocabsize = encoder_vocab_size
        self.tagvocabsize = decoder_vocab_size
        self.comlen = enc_seq_len
        self.taglen = dec_seq_len


    def create_model(self):
        lstm_units = 100

        # Comment input
        encoder_inputs = Input(shape=(self.comlen,))
        encoder_embedding = Embedding(output_dim=100, input_dim=self.comvocabsize, mask_zero=True)(encoder_inputs)
        encoder = LSTM(lstm_units, return_state=True, go_backwards=True)
        encoder_outputs, state_h, state_c= encoder(encoder_embedding)
        encoder_states = [state_h, state_c]
        # Tag input
        decoder_inputs = Input(shape=(self.taglen,))
        decoder_embedding = Embedding(output_dim=100, input_dim=self.tagvocabsize, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.tagvocabsize, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        train_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

        # Define Inference Model
        # Inference Encoder
        encoder_model = Model(encoder_inputs, encoder_states)

        # Inference Decoder
        decoder_state_h = Input(shape=(lstm_units,))
        decoder_state_c = Input(shape=(lstm_units,))
        decoder_state_inputs = [decoder_state_h, decoder_state_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs]+decoder_states)

        return train_model, encoder_model, decoder_model
