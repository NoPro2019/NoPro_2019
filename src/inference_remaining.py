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
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--epoch', type=int, default=None)
parser.add_argument('--best', action='store_true', default=False)
args = parser.parse_args()
gpu = args.gpu
model_type = args.model
epoch = args.epoch
path = args.path
best = args.best

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


def predict(data, encoder, decoder, inf_steps, out_vocab_size, reftok, reverse_word_map, com, fids, outpath, fstuff, src):
    #print(len(data))
    outdict = {}
    total = len(data)
    count = 1
    for k, (inp, fid) in enumerate(zip(data, fids)):
        state = encoder.predict(np.array([inp]))
        
        target_seq = np.zeros((1,1,out_vocab_size))
        target_seq[0, 0, reftok.word_index['<s>']] = 1

        output = []

        for i in range(seqlen):
   
            output_tokens, state_h, state_c = decoder.predict([target_seq]+state)
            idx = np.argmax(output_tokens[0, 0, :])
            if idx == 0:
                sampled_char = '0'
            else:
                sampled_char = reverse_word_map[idx]

            output.append(sampled_char)


            target_seq = np.zeros((1,1,out_vocab_size))
            target_seq[0,0,idx] = 1
            if sampled_char == '</s>':
                break

            state = [state_h, state_c]

        #print(reftok.word_index)
        #print(output)
        # exit()
        fout = []
        selected = []
        for i in range(len(output)):
            if output[i] == '2':
                selected.append(i)
            fout.append(output[i])

        #print(selected)
        #print(com[k])
        #print(ref[k])
        try:
            pred = ' '.join(com[k].split()[selected[0]:selected[-1]+1])
        except:
            pred = ''

        #print("{}---{}".format(fid,pred))
        if selected == []:
            s1 = -1
            s2 = -1
        else:
            s1 = selected[0]
            s2 = selected[-1]+1
        outdict[fid] = (com[k],src[k],pred,s1,s2)
        if count % 1000 == 0:
            print(count,total,float(count)/float(total))
            pickle.dump(outdict, open('../../data/train_remaining.pkl', 'wb'))
        count += 1
        

def predict_batch(data, encoder, decoder, inf_steps, out_vocab_size, reftok, reverse_word_map, com, fids, outpath, fstuff, src):
    #print(len(data))
    data = data[:10]
    outdict = {}
    total = len(data)
    count = 1

    states = encoder.predict(data)
    

    target_seq = np.zeros((1,states[0].shape[0],out_vocab_size))
    target_seq[0, :, reftok.word_index['<s>']] = 1
    print(target_seq)
    output = np.zeros((states[0].shape[0],inf_steps))

    for i in range(seqlen):

        output_tokens, state_h, state_c = decoder.predict([target_seq]+states)

        print(output_tokens)
        idx = [np.argmax(np.argmax(x) for x in output_tokens[0])]
        for j in idx:
            if j == 0:
                sampled_char = '0'
            else:
                sampled_char = reverse_word_map[j]

            output[j,i] = sampled_char


        target_seq = np.zeros((1,1,out_vocab_size))
        print(idx)
        for j in range(len(data)):
            print(i, idx[j])
            target_seq[0,i,idx[j]] = 1

        print(target_seq)
        if sampled_char == '</s>':
            break
        print(target_seq)
        #state = [state_h, state_c]

    # #print(reftok.word_index)
    # #print(output)
    # # exit()
    # fout = []
    # selected = []
    # for i in range(len(output)):
    #     if output[i] == '2':
    #         selected.append(i)
    #     fout.append(output[i])

    # #print(selected)
    # #print(com[k])
    # #print(ref[k])
    # try:
    #     pred = ' '.join(com[k].split()[selected[0]:selected[-1]+1])
    # except:
    #     pred = ''

    # #print("{}---{}".format(fid,pred))
    # if selected == []:
    #     s1 = -1
    #     s2 = -1
    # else:
    #     s1 = selected[0]
    #     s2 = selected[-1]+1
    # outdict[fid] = (com[k],src[k],pred,s1,s2)
    # if count % 1000 == 0:
    #     print(count,total,float(count)/float(total))
    # count += 1
    #pickle.dump(outdict, open('../../data/train_remaining.pkl', 'wb'))



if path is None:
    files = os.listdir('trained/{}/'.format(model_type))
    files.sort()
    filedate = files[-1]
    outpath = 'trained/{}/{}/'.format(model_type, filedate)
    all_files = os.listdir(outpath)
else:
    all_files = os.listdir(path)
    filedate = path.split('/')[-1]
    outpath = 'trained/{}/{}/'.format(model_type, filedate)

# Get best models based on validation accuracy
# 0=modelname, 1=epoch, 2=val_loss, 3=val_acc
files = [x for x in all_files if 'encoder' not in x.lower() and 'decoder' not in x.lower() and x.endswith('.h5')]
if best:
    
    best = ''
    vbest = 0
    ebest = ''
    for f in files:
        tmp = f.split('_')
        va = tmp[3]
        va = int(va.split('.')[1])
        if va > vbest:
            vbest = va
            ebest = tmp[1]
            best = f
    last = 'BEST_'+best
    last_epoch = ebest
else:
    files.sort()
    last = files[-1]
    last_epoch = last.split('_')[1]

files = [x for x in all_files if last_epoch in x and ('encoder' in x or 'decoder' in x)]

files.sort()

for f in files:
    if 'encoder' in f:
        encoder = keras.models.load_model('trained/{}/{}/{}'.format(model_type, filedate,f))
    else:
        decoder = keras.models.load_model('trained/{}/{}/{}'.format(model_type, filedate,f))

print("Predicting On -- {}".format(outpath))
print("--\t{}\t--".format(last))

dataprep = '../../data/remaining_set.pkl'
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
com_vocab_size = 2000
ref_vocab_size = 2000

comments = list(zip(*data))[1]
src = list(zip(*data))[0]


comtok = pickle.load(open('trained/{}/{}/comtok.pkl'.format(model_type, filedate), 'rb'))
reftok = pickle.load(open('trained/{}/{}/reftok.pkl'.format(model_type, filedate), 'rb'))


encoder_in = comtok.texts_to_sequences(comments)

if model_type == 'bilstm-f':
    srctok = pickle.load(open('trained/{}/{}/srctok.pkl'.format(model_type, filedate), 'rb'))
    src_in = srctok.texts_to_sequences(src)
    src_in = pad_sequences(src_in, maxlen=srclen, padding="post", truncating="post")

encoder_in = pad_sequences(encoder_in, maxlen=seqlen, padding="post", truncating="post")

#encoder_in = np.array([encoder_in])

print("--------Testing---------")
print("Encoder Input Shape - {}".format(encoder_in.shape))

#encoder = keras.models.load_model('attendgru_E20_nodecemb_encoder.h5')
#decoder = keras.models.load_model('attendgru_E20_nodecemb_decoder.h5')

reverse_word_map = dict(map(reversed, reftok.word_index.items()))

if model_type == 'bilstm':
    predict(encoder_in, encoder, decoder, targetlen, len(reftok.word_index)+1, reftok, reverse_word_map, comments, fids, outpath, last, src)

elif model_type == 'bilstm-f':
    predict_attendgru(encoder_in, src_in, encoder, decoder, targetlen, len(reftok.word_index)+1, reftok, reverse_word_map, comments, inrefs, fids, outpath, last)