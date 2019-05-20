from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

class VanillaLSTMModel:
    def __init__(self):
        self.comvocabsize = 50000
        self.tagvocabsize = 5
        self.comlen = 100
        self.taglen = 100

    def create_model(self):
        com_input = Input(shape=(self.comlen,))
        tag_input = Input(shape=(self.taglen,))

        xd = Embedding(output_dim=100, input_dim=self.comvocabsize, mask_zero=True)(com_input)
        ld = LSTM(256, return_state=True, activation='tanh')
        ldout, state_h, state_c = ld(xd)

        xc = Embedding(output_dim=100, input_dim=self.tagvocabsize, mask_zero=True)(tag_input)
        lc = LSTM(256)
        lcout = lc(xc, initial_state=[state_h, state_c])

        out = Dense(self.tagvocabsize, activation='softmax')(lcout)

        model = Model(inputs=[com_input, tag_input], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipnorm=1.), metrics=['accuracy'])
        return model
