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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--model', type=str, default='bilstm')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()
gpu = args.gpu
model_type = args.model
test = args.test

seed = 1337
random.seed(seed)
np.random.seed(seed)

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = gpu

def spancom_p(y_true, y_pred):

  y_true = K.argmax(y_true, axis=-1)
  y_pred = K.argmax(y_pred, axis=-1)

  mask = K.cast(K.not_equal(y_true, 0), 'int64')
  matches = K.cast(K.equal(y_true, y_pred), 'int64')*mask
  accuracy = K.sum(matches)/K.maximum(K.sum(mask), 1)
  return accuracy


def spancom_r(y_true, y_pred):

  y_true = K.argmax(y_true, axis=-1)
  y_pred = K.argmax(y_pred, axis=-1)

  mask = K.cast(K.not_equal(y_pred, 0), 'int64')
  matches = K.cast(K.equal(y_true, y_pred), 'int64')*mask
  accuracy = K.sum(matches)/K.maximum(K.sum(mask), 1)
  return accuracy


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
    def __init__(self,encoder_in, decoder_in, decoder_out, batchsize, src_in=None):
        self.encoder_in = encoder_in
        self.decoder_in = decoder_in
        self.decoder_out = decoder_out
        self.batchsize = batchsize
        self.src_in = src_in

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
        #     nd.append(to_categorical(o,num_classes=comwords)) 
        # batch_encoder_in = np.array(nd)

        nd = []
        for o in batch_decoder_in:
            nd.append(to_categorical(o, num_classes=refwords))
        batch_decoder_in = np.array(nd)

        nd = []
        for o in batch_decoder_out:
            nd.append(to_categorical(o, num_classes=refwords))
        batch_decoder_out = np.array(nd)

        ###################################################################
        ###################################################################
        ###################################################################

        if self.src_in is None:
            return [batch_encoder_in, batch_decoder_in], [batch_decoder_out]

        batch_src_in = self.src_in[idx*self.batchsize:(idx+1)*self.batchsize]

        return [batch_encoder_in, batch_src_in, batch_decoder_in], [batch_decoder_out]

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
    steps = 1
else:
    batchsize = 500
    steps = int(len(train_fids)/batchsize)+1
    valsteps = int(len(val_fids)/batchsize)+1

epochs = 30
seqlen = 100
srclen = 100
targetlen = 100
com_vocab_size = 2000
src_vocab_size = 2000
ref_vocab_size = 2000

# Any notes about the model
notes = ""

config = {
    "epochs":epochs,
    "modeltype":model_type,
    "randseed":seed,
    "test":test,
    "seqlen":seqlen,
    "targetlen":targetlen,
    "srclen":srclen,
    "comvocabsize":com_vocab_size,
    "srcvocabsize":src_vocab_size,
    "targetvocabsize":ref_vocab_size,
    "notes":notes

}

outpath = config_to_file(model_type, config)

comments = list(zip(*train_data))[0]
inrefs = list(zip(*train_data))[2]
starts = list(zip(*train_data))[3]
ends = list(zip(*train_data))[4]

refs = []
nr = []
for c,r,s,e in zip(comments, inrefs, starts, ends):
    tmp = np.ones(len(c.split()), dtype='float32')
    tmp[s:e+1] = 2
    nr.append(' '.join([str(int(x)) for x in tmp]))
refs = nr

out = [ x + ' </s>' for x in refs]
refs = ['<s> '+x+ ' </s>' for x in refs]

val_comments = list(zip(*val_data))[0]
val_refs = list(zip(*val_data))[2]
val_starts = list(zip(*val_data))[3]
val_ends = list(zip(*val_data))[4]

nr = []
for c,r,s,e in zip(val_comments, val_refs, val_starts, val_ends):
    tmp = np.ones(len(c.split()), dtype='float32')
    tmp[s:e+1] = 2
    nr.append(' '.join([str(int(x)) for x in tmp]))
val_refs = nr

val_out = [ x + ' </s>' for x in val_refs]
val_refs = ['<s> '+x+' </s>' for x in val_refs]

comtok = Tokenizer(num_words=com_vocab_size, filters='')
comtok.fit_on_texts(comments)
pickle.dump(comtok, open('{}/comtok.pkl'.format(outpath), 'wb'))

reftok = Tokenizer(num_words=ref_vocab_size, filters='')
reftok.fit_on_texts(refs)
reftok.fit_on_texts(out)
pickle.dump(reftok, open('{}/reftok.pkl'.format(outpath), 'wb'))

# Get actual word counts after fitting tokenizers
if comtok.num_words > len(comtok.word_index):
    comwords = len(comtok.word_index)+1
else:
    comwords = comtok.num_words+1

if reftok.num_words > len(reftok.word_index):
    refwords = len(reftok.word_index)+1
else:
    refwords = reftok.num_words+1



encoder_in = comtok.texts_to_sequences(comments)
decoder_in = reftok.texts_to_sequences(refs)
decoder_out = reftok.texts_to_sequences(out)

encoder_in = pad_sequences(encoder_in, maxlen=seqlen, padding="post", truncating="post")
decoder_in = pad_sequences(decoder_in, maxlen=targetlen, padding="post", truncating="post")
decoder_out = pad_sequences(decoder_out, maxlen=targetlen, padding="post", truncating="post")

val_encoder_in = comtok.texts_to_sequences(val_comments)
val_decoder_in = reftok.texts_to_sequences(val_refs)
val_decoder_out = reftok.texts_to_sequences(val_out)

val_encoder_in = pad_sequences(val_encoder_in, maxlen=seqlen, padding="post", truncating="post")
val_decoder_in = pad_sequences(val_decoder_in, maxlen=targetlen, padding="post", truncating="post")
val_decoder_out = pad_sequences(val_decoder_out, maxlen=targetlen, padding="post", truncating="post")

if test:
    encoder_in = np.array([encoder_in[8]])
    decoder_in = np.array([decoder_in[8]])
    decoder_out = np.array([decoder_out[8]])

if model_type == 'bilstm-f' or model_type == 'bilstm-csatt':
    src = list(zip(*train_data))[1]
    val_src = list(zip(*val_data))[1]

    srctok = Tokenizer(num_words=src_vocab_size)
    srctok.fit_on_texts(src)
    pickle.dump(srctok, open('{}/srctok.pkl'.format(outpath), 'wb'))

    if src_vocab_size > len(srctok.word_index):
        srcwords = len(srctok.word_index)+1
    else:
        srcwords = src_vocab_size+1


    src_in = srctok.texts_to_sequences(src)
    val_src_in = srctok.texts_to_sequences(val_src)

    src_in = pad_sequences(src_in, maxlen=srclen, padding="post", truncating="post")
    val_src_in = pad_sequences(val_src_in, maxlen=srclen, padding="post", truncating="post")

    if test:
        src_in = np.array([src_in[8]])
    print("BiLSTM-F Model-----------")
    print("--------Training---------")
    print("Encoder Input Shape - {}".format(encoder_in.shape))
    print("Decoder Input Shape - {}".format(decoder_in.shape))
    print("Source  Input Shape - {}".format(src_in.shape))
    print("Decoder Output Shape- {}".format(decoder_out.shape))
    print("-------Validation--------")
    print("Encoder Input Shape - {}".format(val_encoder_in.shape))
    print("Decoder Input Shape - {}".format(val_decoder_in.shape))
    print("Source  Input Shape - {}".format(val_src_in.shape))
    print("Decoder Output Shape- {}".format(val_decoder_out.shape))

    train = batchgen(encoder_in, decoder_in, decoder_out, batchsize, src_in=src_in)
    val = batchgen(val_encoder_in, val_decoder_in, val_decoder_out, batchsize, src_in=val_src_in)

else:
    print("BiLSTM Model-------------")
    print("--------Training---------")
    print("Encoder Input Shape - {}".format(encoder_in.shape))
    print("Decoder Input Shape - {}".format(decoder_in.shape))
    print("Decoder Output Shape- {}".format(decoder_out.shape))
    print("-------Validation--------")
    print("Encoder Input Shape - {}".format(val_encoder_in.shape))
    print("Decoder Input Shape - {}".format(val_decoder_in.shape))
    print("Decoder Output Shape- {}".format(val_decoder_out.shape))

    train = batchgen(encoder_in, decoder_in, decoder_out, batchsize)
    val = batchgen(val_encoder_in, val_decoder_in, val_decoder_out, batchsize)

if model_type == 'bilstm':
    modelname = 'BiLSTM'
    from models.bilstm import BiLSTM
    model = BiLSTM(comwords, refwords, seqlen, targetlen)
    model, encoder, decoder = model.create_model()
    print(model.summary())
elif model_type == 'bilstm-f':
    modelname = 'BiLSTMF'
    from models.bilstm_f import BiLSTM_F
    model = BiLSTM_F(comwords, len(srctok.word_counts)+1, len(reftok.word_counts)+1, seqlen, srclen, targetlen)
    model, encoder, decoder = model.create_model()
    print(model.summary())
elif model_type == 'bilstm-csatt':
    modelname = 'BiLSTMCSATT'
    from models.bilstm_csatt import BiLSTM_CSAtt
    model = BiLSTM_CSAtt(len(comtok.word_counts)+1, len(srctok.word_counts)+1, len(reftok.word_counts)+1, seqlen, srclen, targetlen)
    model, encoder, decoder = model.create_model()
    print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy', spancom_p, spancom_r])
checkpoint = ModelCheckpoint(outpath+'/'+modelname+'_E{epoch:0>3}_VL{val_loss:.3f}_VA{val_acc:.3f}.h5', monitor='val_loss', save_best_only=False)
codercheckpoint = CheckpointCoders(encoder, decoder, outpath)
stopping = EarlyStopping(monitor='val_spancom_p', patience=15, mode='max')
callbacks = [checkpoint, stopping, codercheckpoint]

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



if test:
    if model_type == 'attendgru':
        print("Decoder")
        print(decoder.summary())
        input_ot = []
        rev = dict(map(reversed, reftok.word_index.items()))
        out = ''
        
        target_seq = np.zeros((1,1,len(reftok.word_index)+1))
        target_seq[0,0,reftok.word_index['<s>']] = 1
        com_out, src_out, state_h, state_c = encoder.predict([encoder_in,src_in])
        state = [state_h, state_c]
        for i in range(seqlen):
            output, state_h, state_c = decoder.predict([com_out, src_out, target_seq]+state)
            
            idx = np.argmax(output[0,0,:])
            try:
                out += rev[idx]+' '
            except:
                out += '0 '
            input_ot.append(np.argmax(target_seq[0,0,:]))
            target_seq = np.zeros((1,1,len(reftok.word_index)+1))
            target_seq[0,0,idx] = 1
            state = [state_h, state_c]
            
            #print(state)


    elif model_type == 'bilstm':
        print("Decoder")
        print(decoder.summary())
        rev = dict(map(reversed, reftok.word_index.items()))
        input_ot = []
        out = ''
        state = encoder.predict(encoder_in)
        target_seq = np.zeros((1,1,len(reftok.word_index)+1))
        target_seq[0,0,reftok.word_index['<s>']] = 1
        for i in range(seqlen):
            output, state_h, state_c = decoder.predict([target_seq]+state)
            idx = np.argmax(output[0,0,:])
            try:
                out += rev[idx]+' '
            except:
                out += '0 '
            input_ot.append(np.argmax(target_seq[0,0,:]))
            target_seq = np.zeros((1,1,len(reftok.word_index)+1))
            target_seq[0,0,idx] = 1
            state = [state_h, state_c]

    print("Comment --")
    print(comments[8])
    print("----------")
    print("Reference Sent")
    print(inrefs[8])
    print("--------")
    print("Reference Tokens")
    print(refs[8])
    print("---------------")
    print("Prediction")
    print(out)
    print("-------------")
    print("Target Seq-----")
    print(input_ot)
    print("---------------")

    exit()