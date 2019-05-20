from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, CuDNNGRU, LSTM, CuDNNLSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, TimeDistributed, Bidirectional, dot, add
from keras.optimizers import RMSprop
import keras
import tensorflow as tf
from keras.regularizers import L1L2
from keras.initializers import VarianceScaling

class Commenter:
    def __init__(self, encoder_vocab_size, decoder_vocab_size, enc_seq_len, dec_seq_len):
        self.srcvocabsize = encoder_vocab_size
        self.comvocabsize = decoder_vocab_size
        self.srclen = enc_seq_len
        self.comlen = dec_seq_len


    def create_model(self):
        lstm_units = 100
        embedding_dims = 200
        # Source Code input
        encoder_inputs = Input(shape=(self.srclen,))
        encoder_embedding = Embedding(output_dim=embedding_dims, input_dim=self.srcvocabsize, mask_zero=False)(encoder_inputs)
        encoder = CuDNNLSTM(lstm_units, return_state=True, return_sequences=True)
        encoder_outputs, state_h, state_c = encoder(encoder_embedding)

        encoder_states = [state_h, state_c]

        # Com Input
        decoder_inputs = Input(shape=(self.comlen,))
        decoder_embedding = Embedding(output_dim=embedding_dims, input_dim=self.comvocabsize, mask_zero=False)(decoder_inputs)
        decoder_lstm = CuDNNLSTM(lstm_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        # Src Attention
        src_att = dot([decoder_outputs, encoder_outputs], axes=[2,2])
        src_att = Activation('softmax')(src_att)
        

        # Context Vector
        context = dot([src_att, encoder_outputs], axes=[2,1])
        context = concatenate([context, decoder_outputs])

        decoder_dense = Dense(self.comvocabsize, activation='softmax')
        decoder_outputs = decoder_dense(context)
        train_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)


        # Define Inference Model
        # Inference Encoder
        # encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

        # # Inference Decoder
        # #decoder_state_h = Input(shape=(lstm_units,))
        # #decoder_state_c = Input(shape=(lstm_units,))
        # #decoder_src_inputs = Input(shape=(self.srclen, lstm_units))

        # #decoder_state_h = Input(shape=(lstm_units,))
        # #decoder_state_c = Input(shape=(lstm_units,))
        # decoder_state_inputs = [decoder_state_h, decoder_state_c]
        # #decoder_state_inputs = [decoder_state_h, decoder_state_c]
        # decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_state_inputs)
        # decoder_states = [state_h, state_c]

        # # Decoder Attention
        # #decoder_att = dot([decoder_outputs, decoder_src_inputs], axes=[2,2])
        # #decoder_att = Activation('softmax')(decoder_att)
        
        # # Decoder Context
        # #decoder_context = dot([decoder_att, decoder_src_inputs], axes=[2,1])
        # #decoder_context = concatenate([decoder_context, decoder_outputs])
        
        # decoder_outputs = decoder_dense(decoder_outputs)
        # decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

        return train_model, None, None

if __name__ == "__main__":
    test = Commenter(100, 100, 100, 100)
    train, encoder, decoder = test.create_model()

    print(train.summary())
    print(encoder.summary())
    print(decoder.summary())