import random
import pickle
import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback, EarlyStopping
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--best', action='store_true', default=False)
parser.add_argument('--ref', action='store_true', default=False)
args = parser.parse_args()
gpu = args.gpu
model_type = args.model
epoch = args.epoch
best = args.best
ref = args.ref

random.seed(1337)
np.random.seed(1337)

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = gpu

set_session(tf.Session(config=config))

# Lets try a seq2seq model
class batchgen(keras.utils.Sequence):
    def __init__(self,encoder_in, decoder_in, decoder_out, batchsize):
        self.encoder_in = encoder_in
        self.decoder_in = decoder_in
        self.decoder_out = decoder_out
        self.batchsize = batchsize

    def __len__(self):
        return int(np.ceil(len(self.encoder_in)/float(self.batchsize)))

    def __getitem__(self,idx):

        batch_encoder_in = self.encoder_in[idx*self.batchsize:(idx+1)*self.batchsize]
        batch_decoder_in = self.decoder_in[idx*self.batchsize:(idx+1)*self.batchsize]
        batch_decoder_out = self.decoder_out[idx*self.batchsize:(idx+1)*self.batchsize]
       

        return [batch_encoder_in, batch_decoder_in], [batch_decoder_out]


def predict(data, model, inf_steps, out_vocab_size, comtok, reverse_word_map, com, fids):
    #print(len(data))
    for k, (inp, fid) in enumerate(zip(data[0], fids)):
        inp = np.array([inp])
        out = ''
        target_seq = np.zeros((1,inf_steps))
        target_seq[0,0] = comtok.word_index['<s>']
        for i in range(inf_steps-1):
            output = model.predict([inp, target_seq])

            idx = np.argmax(output[0,i,:])
            try:
                out += reverse_word_map[idx]+' '
            except:
                out += '0 '

            target_seq[0,i+1] = idx
            if idx == comtok.word_index['</s>']:
                break

        #print("{}---{}".format(ref[k],pred))
        print("{},{}".format(fid,out))


model = keras.models.load_model(model_type)

print("Predicting On -- {}".format(model_type))

dataprep = '../../data/testing.pkl'
all_seqs = pickle.load(open(dataprep, "rb"))

# Get data all situates
data = list(all_seqs.items())
#random.shuffle(data)

fids, data = list(zip(*data))

# CONFIGS
batchsize = 30
steps = int(len(fids)/batchsize)+1
epochs = 1
seqlen = 100
srclen = 100
targetlen = 100
oov_token = "<unk>"
com_vocab_size = 10000
src_vocab_size = 10000

if ref:
    comments = list(zip(*data))[2]
else:
    comments = list(zip(*data))[0]

src = list(zip(*data))[1]


decoder_out = [ x + ' </s>' for x in comments]
decoder_in = ['<s> '+x+ ' </s>' for x in comments]

comtok = pickle.load(open('comment_comtok.pkl', 'rb'))
srctok = pickle.load(open('comment_srctok.pkl', 'rb'))

encoder_in = comtok.texts_to_sequences(comments)
src_in = srctok.texts_to_sequences(src)

encoder_in = pad_sequences(encoder_in, maxlen=seqlen, padding="post", truncating="post")
src_in = pad_sequences(src_in, maxlen=srclen, padding="post", truncating="post")

encoder_in = np.array([encoder_in])

print("--------Testing---------")
print("Encoder Input Shape - {}".format(encoder_in.shape))

#encoder = keras.models.load_model('attendgru_E20_nodecemb_encoder.h5')
#decoder = keras.models.load_model('attendgru_E20_nodecemb_decoder.h5')

reverse_word_map = dict(map(reversed, comtok.word_index.items()))

predict(encoder_in, model, targetlen, com_vocab_size, comtok, reverse_word_map, comments, fids)
