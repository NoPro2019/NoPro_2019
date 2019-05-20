from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, CuDNNLSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten, TimeDistributed, Bidirectional, dot, add, RepeatVector
from keras.optimizers import RMSprop
import keras
import tensorflow as tf


class BiLSTM_CSAtt:
    def __init__(
        self, 
        encoder_vocab_size, 
        encoder_src_vocab_size, 
        decoder_vocab_size, 
        enc_seq_len,
        enc_src_len, 
        dec_seq_len
    ):

        self.comvocabsize = encoder_vocab_size
        self.srcvocabsize = encoder_src_vocab_size
        self.tagvocabsize = decoder_vocab_size
        self.comlen = enc_seq_len
        self.srclen = enc_src_len
        self.taglen = dec_seq_len

    
    def create_model(self):
        recurrent_dims = 100
        embedding_dims = 300

        # Comment encoder
        encoder_com_input = Input(shape=(self.comlen,))
        encoder_com_embedding = Embedding(output_dim=embedding_dims, input_dim=self.comvocabsize, mask_zero=False)(encoder_com_input)
        encoder_com = Bidirectional(CuDNNLSTM(recurrent_dims, return_state=True, return_sequences=True))
        encoder_com_output, forward_com_state_h, forward_com_state_c, backward_com_state_h, backward_com_state_c = encoder_com(encoder_com_embedding)
        com_state_h = concatenate([forward_com_state_h, backward_com_state_h])
        com_state_c = concatenate([forward_com_state_c, backward_com_state_c])
        encoder_com_states = [com_state_h, com_state_c]

        # Source code encoder
        encoder_src_input = Input(shape=(self.srclen,))
        encoder_src_embedding = Embedding(output_dim=embedding_dims, input_dim=self.srcvocabsize, mask_zero=False)(encoder_src_input)
        encoder_src = Bidirectional(CuDNNLSTM(recurrent_dims, return_state=True, return_sequences=True))
        encoder_src_output, forward_src_state_h, forward_src_state_c, backward_src_state_h, backward_src_state_c = encoder_src(encoder_src_embedding)
        src_state_h = concatenate([forward_src_state_h, backward_src_state_h])
        src_state_c = concatenate([forward_src_state_c, backward_src_state_c])
        encoder_src_states = [src_state_h, src_state_c]

        # Decoder - we use the encoder_com_states to initialize the decoder LSTM
        decoder_input = Input(shape=(self.taglen, ))
        decoder_embedding = Embedding(output_dim=embedding_dims, input_dim=self.tagvocabsize, mask_zero=False)(decoder_input)
        decoder = CuDNNLSTM(recurrent_dims*2, return_state=True, return_sequences=True)
        decoder_output, _, _ = decoder(decoder_embedding, initial_state=encoder_com_states)

        # Comment Attention
        com_attn = dot([encoder_src_output, encoder_com_output], axes=[2, 2])
        com_attn = Activation('softmax')(com_attn)
        com_context = dot([com_attn, encoder_com_output], axes=[2,1])

        # combine attention and decoder input vectors
        context = concatenate([encoder_com_output, decoder_output, encoder_src_output])
        context_td = TimeDistributed(Dense(300, activation="relu"))
        context = context_td(context)

        #context = Flatten()(context)

        # Training model output
        decoder_dense = Dense(self.tagvocabsize, activation="softmax")
        decoder_output = decoder_dense(context)
        train_model = Model(inputs=[encoder_com_input, encoder_src_input, decoder_input], outputs=decoder_output)

        # Define Inference Model Below
        # Inference Encoder - We only need this to get the initial encoder states to pass to the decoder
        encoder_model = Model(inputs=[encoder_com_input, encoder_src_input], outputs=[encoder_com_output, encoder_src_output]+encoder_com_states)
        

        # Inference Decoder
        # This model will take in all of our inputs (comment, src, and decoded-so-far) and the initial decoder LSTM states.
        decoder_state_h = Input(shape=(recurrent_dims*2,))
        decoder_state_c = Input(shape=(recurrent_dims*2,))
        
        encoder_com_in = Input(shape=(self.comlen,recurrent_dims*2))
        encoder_src_in = Input(shape=(self.srclen,recurrent_dims*2))


        decoder_state_inputs = [decoder_state_h, decoder_state_c]
        decoder_output, state_h, state_c = decoder(decoder_embedding, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]

        # Comment Attention
        com_attn = dot([decoder_output, encoder_com_output], axes=[2, 2])
        com_attn = Activation('softmax')(com_attn)
        com_context = dot([com_attn, encoder_com_output], axes=[2,1])


        new_context = concatenate([com_context, decoder_output])
        new_context = context_td(new_context)
        decoder_output = decoder_dense(new_context)
        decoder_model = Model([encoder_com_in, encoder_src_in, decoder_input]+decoder_state_inputs, [decoder_output] + decoder_states)

        return train_model, encoder_model, decoder_model