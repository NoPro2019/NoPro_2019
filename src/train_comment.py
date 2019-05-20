import random
import pickle
import MySQLdb
import sys
import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback, EarlyStopping
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import argparse
from keras.utils import plot_model
import os
import datetime
from keras import backend as K
import tensorflow as tf
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--ref', action='store_true', default=False)
args = parser.parse_args()
gpu = args.gpu
test = args.test
ref = args.ref

if ref:
    model_type='commenter-ref'


seed = 1337
random.seed(seed)
np.random.seed(seed)

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = gpu



def config_to_file(modelname, config):
    test = config['test']
    if not test:
        path = 'trained/{}/{date:%m_%d_%H_%M}/'.format(modelname, date=datetime.datetime.now())
        os.makedirs(path)
    else:
        path = 'test/{}/{date:%m_%d_%H_%M}/'.format(modelname, date=datetime.datetime.now())
        os.makedirs(path, exist_ok=True)
    fo = open(path+'config', 'w+')
    for key, value in config.items():
        fo.write("{}:{}\n".format(key, value))
    fo.close()
    return path

set_session(tf.Session(config=config))

class CheckpointCoders(keras.callbacks.Callback):
    def __init__(self, encoder, decoder, path, verbose=False):
        self.encoder = encoder
        self.decoder = decoder
        self.path = path
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            print("Epoch {} - Saving encoder/decoder to {}".format(epoch+1, path))

        self.encoder.save(self.path+"encoder_E{:0>3}_VA{:.3f}_VL{:.3f}.h5".format(epoch+1, logs.get('val_acc'), logs.get('val_loss')), overwrite=True)
        self.decoder.save(self.path+"decoder_E{:0>3}_VA{:.3f}_VL{:.3f}.h5".format(epoch+1, logs.get('val_acc'), logs.get('val_loss')), overwrite=True)

class batchgen(keras.utils.Sequence):
    def __init__(self,encoder_in, decoder_in, decoder_out, batchsize, comtok):
        self.encoder_in = encoder_in
        self.decoder_in = decoder_in
        self.decoder_out = decoder_out
        self.comtok = comtok
        self.batchsize = batchsize

    def __len__(self):
        return int(np.ceil(len(self.encoder_in)/float(self.batchsize)))

    def __getitem__(self,idx):

        batch_encoder_in = self.encoder_in[idx*self.batchsize:(idx+1)*self.batchsize]
        batch_decoder_in = self.decoder_in[idx*self.batchsize:(idx+1)*self.batchsize]
        batch_decoder_out = self.decoder_out[idx*self.batchsize:(idx+1)*self.batchsize]
       
        ####################################################################
        ##These will be commented if not using an embedding for that input##
        ####################################################################

        # nd = []
        # for o in batch_encoder_in:
        #     nd.append(to_categorical(o,num_classes=len(comtok.word_counts)+1)) 
        # batch_encoder_in = np.array(nd)

        # nd = []
        # for o in batch_decoder_in:
        #     nd.append(to_categorical(o, num_classes=len(reftok.word_counts)+1))
        # batch_decoder_in = np.array(nd)

        nd = []
        for o in batch_decoder_out:
            nd.append(to_categorical(o, num_classes=self.comtok.num_words))
        batch_decoder_out = np.array(nd)

        ###################################################################
        ###################################################################
        ###################################################################

        return [batch_encoder_in, batch_decoder_in], [batch_decoder_out]

dataprep = '../../data/training.pkl'
all_seqs = pickle.load(open(dataprep, "rb"))

# Randomly shuffle data so we don't have sequential functions in training/validation
data = list(all_seqs.items())


random.seed(seed)
if not test:
    random.shuffle(data)

train_data = data[:int(len(data)*.9)]
val_data = data[int(len(data)*.9):]

train_fids, train_data = list(zip(*train_data))
val_fids, val_data = list(zip(*val_data))

# CONFIGS
if test:
    batchsize = 1
    steps = 10
else:
    batchsize = 100
    steps = int(len(train_fids)/batchsize)+1
    valsteps = int(len(val_fids)/batchsize)+1

epochs = 100
comlen = 100
srclen = 100
com_vocab_size = 10000
src_vocab_size = 10000


# Any notes about the model
notes = "Trained on ref comments"

config = {
    "epochs":epochs,
    "modeltype":model_type,
    "randseed":seed,
    "test":test,
    "comlen":comlen,
    "srclen":srclen,
    "comvocabsize":com_vocab_size,
    "srcvocabsize":src_vocab_size,
    "notes":notes

}

outpath = config_to_file(model_type, config)

if ref:
    comments = list(zip(*train_data))[2]
else:
    comments = list(zip(*train_data))[0]
src = list(zip(*train_data))[1]

decoder_out = [x+' </s>' for x in comments]
decoder_in = ['<s> '+x+' </s>' for x in comments]

val_comments = list(zip(*val_data))[2]
val_src = list(zip(*val_data))[1]

val_decoder_out = [x+' </s>' for x in val_comments]
val_decoder_in = ['<s> '+x+' </s>' for x in val_comments]

# nc = []
# for com in comments:
#     nc.append(' '.join(word_tokenize(com)))
# com = nc

# ns = []
# for s in src:
#     ns.append(' '.join(word_tokenize(s)))
# src = ns

srctok = Tokenizer(num_words=src_vocab_size, filters='')
srctok.fit_on_texts(src)
pickle.dump(srctok, open('comment_srctok.pkl', 'wb'))

comtok = Tokenizer(num_words=com_vocab_size, filters='')
comtok.fit_on_texts(decoder_in)
pickle.dump(comtok, open('comment_comtok.pkl', 'wb'))


encoder_in = srctok.texts_to_sequences(src)
decoder_in = comtok.texts_to_sequences(decoder_in)
decoder_out = comtok.texts_to_sequences(decoder_out)

encoder_in = pad_sequences(encoder_in, maxlen=srclen, padding="post", truncating="post")
decoder_in = pad_sequences(decoder_in, maxlen=comlen, padding="post", truncating="post")
decoder_out = pad_sequences(decoder_out, maxlen=comlen, padding="post", truncating="post")

val_encoder_in = srctok.texts_to_sequences(val_comments)
val_decoder_in = comtok.texts_to_sequences(val_decoder_in)
val_decoder_out = comtok.texts_to_sequences(val_decoder_out)

val_encoder_in = pad_sequences(val_encoder_in, maxlen=srclen, padding="post", truncating="post")
val_decoder_in = pad_sequences(val_decoder_in, maxlen=comlen, padding="post", truncating="post")
val_decoder_out = pad_sequences(val_decoder_out, maxlen=comlen, padding="post", truncating="post")

if test:
    comments = comments[8]
    src = src[8]
    encoder_in = np.array([encoder_in[8]])
    decoder_in = np.array([decoder_in[8]])
    decoder_out = np.array([decoder_out[8]])



print("Commenter Model-------------")
print("--------Training---------")
print("Encoder Input Shape - {}".format(encoder_in.shape))
print("Decoder Input Shape - {}".format(decoder_in.shape))
print("Decoder Output Shape- {}".format(decoder_out.shape))
print("-------Validation--------")
print("Encoder Input Shape - {}".format(val_encoder_in.shape))
print("Decoder Input Shape - {}".format(val_decoder_in.shape))
print("Decoder Output Shape- {}".format(val_decoder_out.shape))

train = batchgen(encoder_in, decoder_in, decoder_out, batchsize, comtok)
val = batchgen(val_encoder_in, val_decoder_in, val_decoder_out, batchsize, comtok)

if ref:
    modelname = 'commenter-ref'
else:
    modelname = 'commenter'
from models.commenter import Commenter
model = Commenter(src_vocab_size, com_vocab_size, srclen, comlen)
model, encoder, decoder = model.create_model()



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
print(model.summary())
checkpoint = ModelCheckpoint(outpath+'/'+modelname+'_E{epoch:0>3}_VL{val_loss:.3f}_VA{val_acc:.3f}.h5', monitor='val_loss', save_best_only=False)
codercheckpoint = CheckpointCoders(encoder, decoder, outpath)
stopping = EarlyStopping(monitor='val_spancom_p', patience=10, mode='max')
callbacks = [checkpoint]

if test:
    model.fit_generator(train,
                        epochs=epochs, 
                        steps_per_epoch=steps, 
                        verbose=1, 
                        workers=8,
                        max_queue_size=24,
                        shuffle=True)
else:
    model.fit_generator(train,
                    validation_data=val,
                    validation_steps=valsteps,
                    epochs=epochs, 
                    steps_per_epoch=steps, 
                    verbose=1, 
                    callbacks=callbacks, 
                    workers=8,
                    max_queue_size=24,
                    shuffle=True)


plot_model(model, to_file=outpath+'train_model.png')
plot_model(encoder, to_file=outpath+'encoder.png')
plot_model(decoder, to_file=outpath+'decoder.png')

srcrev = dict(map(reversed, srctok.word_index.items()))
rev = dict(map(reversed, comtok.word_index.items()))
if test:
    out = ''
    target_seq = np.zeros((1,comlen))
    target_seq[0,0] = comtok.word_index['<s>']
    for i in range(comlen-1):
        output = model.predict([encoder_in,target_seq])

        idx = np.argmax(output[0,i,:])

        try:
            out += rev[idx]+' '
        except:
            out += '0 '

        target_seq[0,i+1] = idx



    # print("Decoder")
    # print(decoder.summary())
    # input_ot = []
    # 
    # out = ''
    
    # target_seq = np.zeros((1,comlen))
    # target_seq[0,0] = comtok.word_index['<s>']
    # enc_out, state_h, state_c = encoder.predict(encoder_in)
    # states = [state_h, state_c]
    # for i in range(comlen-1):
    #     print(target_seq.shape)
    #     output, state_h, state_c = decoder.predict([target_seq] + states)
    #     idx = np.argmax(output[0,i,:])
    #     try:
    #         out += rev[idx]+' '
    #     except:
    #         out += '0 '

    #     target_seq[0,i+1] = idx
    #     if i == 4:
    #         break
    #     states = [state_h, state_c]
    print(' '.join(rev[x] for x in target_seq[0] if x != 0))
    print('--------------')
    print(' '.join([rev[x] for x in decoder_in[0] if x != 0]))
    print('--------------')
    print(' '.join([rev[x] for x in decoder_out[0] if x != 0]))
    print('--------------')
    print(' '.join([srcrev[x] for x in encoder_in[0] if x != 0]))
    print('-------------')
    print(out)

    # exit()