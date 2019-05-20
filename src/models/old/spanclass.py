from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, TimeDistributed
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
        com_input = Input(shape=(self.comlen,), name='input_1')
        tag_input = Input(shape=(self.taglen,), name='input_2')

        xd = Embedding(output_dim=100, input_dim=self.comvocabsize, mask_zero=True)(com_input)
        ld= LSTM(100, return_state=True, return_sequences=False, activation='tanh')
        ldout, e1, e2 = ld(xd)

        xc = Embedding(output_dim=100, input_dim=self.tagvocabsize, mask_zero=True)(tag_input)
        decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
        dout, _, _ = decoder_lstm(xc, initial_state=[e1, e2])
        #out = concatenate([ld,xc])

        #out = concatenate([ldout, lcout])
        out = Dense(self.tagvocabsize, activation='softmax', name='out')(dout)

        model = Model(inputs=[com_input, tag_input], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model
