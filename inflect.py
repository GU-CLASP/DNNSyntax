from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, AtrousConvolution1D, ZeroPadding1D
from keras.layers import Dense, TimeDistributed, Input, Flatten, Reshape, Embedding, LSTM, GRU, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.utils import np_utils
import keras
import sys
import numpy as np
import os.path as op
import os as os
import os
import math
import random
import json


os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


# Constants
## Network parameters
embedding_size = 50
lstm_memory_size = 1350
MAXLEN = 50
batch_size = 1024
rnn_cell = LSTM
# rnn_cell = GRU
vocabulary_size = 10000
arch_type = "rnn"
dropout = 0.1
rnn_layers = 2
# arch_type = "cnn"

## data parameters
configType = 'wacky'

root_dir = "/scratch/" + configType + str(vocabulary_size)

model_ident = configType + "." + (rnn_cell.__name__ if arch_type == 'rnn' else arch_type) + str(lstm_memory_size) 

train_set = op.join(root_dir, "verbsTrain")
test_set = op.join(root_dir, "verbsTest")
vocab_file   = op.join(root_dir, "vocab")
model_root_dir = op.join (root_dir, "modelsInfl" + model_ident)
checkpoint_path = op.join (model_root_dir, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
model_path = op.join (model_root_dir, "final")

print ("Training and checkpointing at ", model_root_dir)

os.makedirs(model_root_dir, exist_ok=True)

def pad(ws):
    return ([0]*(MAXLEN - len(ws)) + ws)


def parse_inflect(ls):
    for l in ls:
        (attractors,isPlural,subj_index,verb_index,sent) = l.split('\t')
        vi = int(verb_index)-1
        s = [int(w) for w in sent.split()]
        verb = s[vi]
        # if (isSingularWord(verb) | isPluralWord(verb)):
            # print("all:", decoder(s), "cut:", decoder(s[:vi]), "v = ", decoder([verb]))
        yield {'sent': np.array(pad(s[:vi])),
               'attractors': int(attractors),
               'verb' : verb,
               'plural': int(isPlural)}
         # else: print("unknown verb:", decoder([verb]), "in:", decoder(s))

def make_input(examples):
    return np.array([seq["sent"] for seq in examples])

def make_output(examples):
    return np.array([seq["plural"] for seq in examples])

def load_examples(fname, shuffle = True):
    global codes
    examples = list(parse_inflect(open(fname).readlines()))
    print ("Loaded #", len(examples))
    return (make_input(examples), make_output(examples), np.array([seq["attractors"] for seq in examples]))

def load_vocab(vocab_filename):
    print ("Loading vocabulary: ", vocab_filename)
    vocab = {'<PAD>':0}
    rev =  {0:'<PAD>'}
    for l in open(vocab_filename, encoding = "latin-1").readlines():
       wd, idx = l.split()
       x = int(idx)
       vocab[wd] = x
       rev[x] = wd
    return (vocab,rev,1+x)

print ("Loading dictionary " + vocab_file)

(vocab,codes,n_vocab) = load_vocab(vocab_file)

filter_vocab = lambda ws: [w for w in ws if w in vocab]

if configType == 'linzen':
    singular_toks = filter_vocab(["'s", "is","has","VBZ"])
    plural_toks = filter_vocab(["are","have","VBP","state"])
    # unknown verb: ['i', 'and', "years", "later", "were", "up", "here", etc.]
elif configType == 'wacky':
    singular_toks = filter_vocab(["is","has","VVZ", "'s", "states", "VHZ"])
    plural_toks = filter_vocab(["are","have","VVP", "VBP", "VHP", "state"])
    # unknown verb: ['became']
    # unknown verb: ['all']
    # unknown verb: ['born']
    # unknown verb: ['at']
else:
    abort

def isPluralWord(w): return int(w in plurals)
def isSingularWord(w): return int(w in singulars)

decoder = lambda words: [codes[w] for w in words]

singulars = [vocab[x] for x in singular_toks]
plurals   = [vocab[x] for x in plural_toks]
encoder = lambda words: [vocab[w] for w in words]

print ("Size of vocabulary: " + str(n_vocab))

def print_model(model):
   print ("MODEL:")
   print (model.to_yaml())

commands = sys.argv[1:]
def pop():
    global commands
    ret = commands[0]
    commands = commands[1:]
    return ret


while len(commands) > 0:
    cmd = pop()
    if cmd == 'load':
        filepath = pop()
        print ("Loading and compiling model: " + filepath)
        language_model = keras.models.load_model(filepath)
        print_model(language_model)
    elif cmd == 'create':
        if arch_type == 'rnn':
            language_model = Sequential([
                Embedding(n_vocab, embedding_size, input_length=MAXLEN)] +
                [rnn_cell(lstm_memory_size, return_sequences=i!=rnn_layers-1, consume_less='gpu',
                          unroll = True, dropout_U=dropout, dropout_W=dropout)
                 for i in range(rnn_layers)] + 
                [Dense(1, activation='sigmoid')])
        elif arch_type == 'cnn':
            language_model = Sequential()
            language_model.add(Embedding(input_dim = n_vocab,
                                         output_dim = embedding_size,
                                         input_length = MAXLEN))
            act = 'relu' # keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
            language_model.add(      Convolution1D(20, 7, activation=act,               border_mode='same'))
            language_model.add(AtrousConvolution1D(15, 5, activation=act,atrous_rate=2, border_mode='same'))
            language_model.add(AtrousConvolution1D(10, 5, activation=act,atrous_rate=4, border_mode='same'))
            language_model.add(AtrousConvolution1D(5,  3, activation=act,atrous_rate=6, border_mode='same'))
            language_model.add(Flatten()) # ??? why ???
            language_model.add(Dropout(0.5))
            language_model.add(Dense(output_dim = 1, activation = 'sigmoid'))
        else: abort

        language_model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
        print_model(language_model)
    elif cmd == 'train':
        print ("Loading examples")
        (x,y,_) = load_examples(train_set)
        language_model.fit(x,y,
                           validation_split=0.1,
                           batch_size = batch_size,
                           nb_epoch=50,
                           shuffle = True,
                           callbacks = [EarlyStopping(monitor='val_loss'),
                                        ModelCheckpoint(checkpoint_path)])
        language_model.save(model_path)
    elif cmd == 'test':
        print ("Loading test set " + test_set)
        (x,y,nattr) = load_examples(test_set)
        print ("Testing on ", len(x))
        y_ = language_model.predict_classes(x,batch_size = batch_size)
        # print ("x.shape() =", x.shape(), "y.shape() = ", y.shape())
        correct = {}
        total = {}
        for i in range(50):
            correct[i] = 0
            total[i] = 0
        for i in range(len(x)):
            correct[nattr[i]] += int(y[i] == y_[i])
            total[nattr[i]] += 1
        print ("|  attractors | correct | total | accuracy |  ")
        print ("|-")
        for k in total.keys():
            if total[k] > 0:
                print ("|", k, "|", correct[k], "|",total[k], "|", correct[k] / float (total[k]), "|")
    elif cmd == "viz":
        import viz
        sentence = pop().split()
        output = []
        get_viz_output = keras.backend.function([language_model.layers[0].input],
                                                [language_model.layers[2].output])
        print ("Evaluating ", sentence)
        for i in range(len(sentence)+1):
            print("Before", sentence[i])
            x = np.array([pad(encoder(sentence[:i]))])
            output.append(get_viz_output([x])[0][0])
        fname = "-".join(sentence) + "." + model_ident
        jfile = open(fname + ".json", "w")
        # json.dump((np.array(output).tolist()), jfile)
        viz.gen_heatmap([],["<PAD>"] + sentence,np.array(output),fname)
    elif cmd == "predict":
        sentence = pop().split()
        encoded_s = encoder(sentence)
        x = np.array([pad(encoder(sentence[:i])) for i in range(len(sentence)+1)])
        y_ = language_model.predict_classes(x,batch_size = batch_size)
        print ("|  len | wd | encoded | prediction | singular  | plural | " )
        print ("|-")
        for i in range(len(sentence)):
            print ("|", i, "|", sentence[i], "|", encoded_s[i], "|", y_[i] , "|", isSingularWord(encoded_s[i]), "|", isPluralWord(encoded_s[i]))
    else:
        print ('Unknown command: ' + cmd)




