from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, CuDNNLSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, TimeDistributed, Bidirectional, dot, add
from keras.optimizers import RMSprop
import keras
import tensorflow as tf
from keras.regularizers import L1L2

class Seq2SeqBILSTM:
    def __init__(self, encoder_vocab_size, decoder_vocab_size, enc_seq_len, dec_seq_len):
        self.comvocabsize = encoder_vocab_size
        self.tagvocabsize = decoder_vocab_size
        self.comlen = enc_seq_len
        self.taglen = dec_seq_len


    def create_model(self):
        lstm_units = 100

        # Comment input
        encoder_inputs = Input(shape=(self.comlen,))
        encoder_embedding = Embedding(output_dim=10, input_dim=self.comvocabsize, mask_zero=False)(encoder_inputs)
        encoder = Bidirectional(CuDNNLSTM(lstm_units, return_state=True, bias_regularizer=L1L2(0.01, 0.0)))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_embedding)
        state_h = concatenate([forward_h, backward_h])
        state_c = concatenate([forward_c, backward_c])

        encoder_states = [state_h, state_c]

        # Tag input
        decoder_inputs = Input(shape=(self.taglen,))
        decoder_embedding = Embedding(output_dim=2, input_dim=self.tagvocabsize, mask_zero=False)(decoder_inputs)
        decoder_lstm = CuDNNLSTM(lstm_units*2, return_sequences=True, return_state=True, bias_regularizer=L1L2(0.01, 0.0))
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        #decoder_dropout = Dropout(0.5)(decoder_outputs)
        #decoder_d = TimeDistributed(Dense(400))(decoder_dropout)
        decoder_dense = Dense(self.tagvocabsize, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        train_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

        # Define Inference Model
        # Inference Encoder
        encoder_model = Model(encoder_inputs, encoder_states)

        # Inference Decoder
        decoder_state_h = Input(shape=(lstm_units*2,))
        decoder_state_c = Input(shape=(lstm_units*2,))
        decoder_state_inputs = [decoder_state_h, decoder_state_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_state_inputs)
        #decoder_dropout = Dropout(0.5)(decoder_outputs)
        #decoder_d = TimeDistributed(Dense(400))(decoder_dropout)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs]+decoder_states)

        return train_model, encoder_model, decoder_model
