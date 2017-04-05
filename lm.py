from keras.layers.embeddings import Embedding
from keras.layers import Dense, TimeDistributed, Input, Flatten, Reshape, Embedding, LSTM, GRU
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.utils import np_utils
import keras
import sys
import numpy as np
import os.path as op
import os as os
import os
import math

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"


# Constants
## Network parameters
embedding_size = 50
lstm_memory_size = 1200
MAXLEN = 50
batch_size = 1024

## data parameters
configType = 'wacky'
if configType == 'linzen':
    root_dir = "/scratch/linzen100/"
    singular_toks = ["is","has","VBZ"]
    plural_toks = ["are","have","VBP"]
elif configType == 'wacky':
    root_dir = "/scratch/wacky100/"
    singular_toks = {"is":"are",
                     "has":"have",
                     "VVZ":"VVP",
                     "'s":"are",
                     "states":"state",
                     "VHZ":"VHP"}
    plural_toks = {"are":"is",
                   "have":"has",
                   "VVP":"VVZ", "VBP":"is", "VHP":"VHZ",
                   "state":"states"}
else:
    abort

train_set = [op.join(root_dir, "train")]
val_set = [op.join(root_dir, "val")]
test_set = [op.join(root_dir, "test")]
vocab_file   = op.join(root_dir, "vocab")
model_root_dir = op.join (root_dir, "models" + str(lstm_memory_size))
checkpoint_path = op.join (model_root_dir, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
model_path = op.join (model_root_dir, "final")

os.makedirs(model_root_dir, exist_ok=True)

def pad(ws):
    return (ws + [0]*(MAXLEN - len(ws)))

parse_sentence = lambda l: pad([int(w) for w in l.split()])

def parse_examples(lines): return [parse_sentence(l) for l in lines]

def make_input(examples):
    return np.array([seq for seq in examples])

def make_output(examples):
    return np.array([np_utils.to_categorical(np.array(seq[1:] + [0]),nb_classes=n_vocab) for seq in examples])

def load_vocab(vocab_filename):
    vocab = {'<PAD>':0}
    for l in open(vocab_filename).readlines():
       wd, idx = l.split()
       x = int(idx)
       vocab[wd] = x
    return (vocab,1+x)

# Preprocessing functions
def sequence_generator(corpus_files):
    lines = []
    while True:
        #ATTN: if this function provides less data then necessary then the program will crash.
        for fname in corpus_files:
            for line in open(fname):
                lines.append(line)
                if len(lines) >= batch_size:
                    examples = parse_examples(lines)
                    # print ("Yield?" + str(make_output(examples).shape))
                    yield (make_input(examples), make_output(examples))
                    lines = []


print ("Loading dictionary " + vocab_file)
(vocab,n_vocab) = load_vocab(vocab_file)

indices_words = dict((vocab[w], w) for w in vocab.keys())

singulars = dict([(vocab[k], vocab[singular_toks[k]]) for k in singular_toks.keys()])
plurals = dict([(vocab[k], vocab[plural_toks[k]]) for k in plural_toks.keys()])

def isInflectibe(w): return (w in plurals) or (w in singulars)
def invertPlurality(w):
    if w in plurals:
        return plurals[w]
    elif w in singulars:
        return singulars[w]
    else:
        error

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


def isPluralWord(w): return w in plurals
def isSingularWord(w): return w in singulars
def isPluralPrediction(arr):
    return sum([arr[i] for i in singulars]) < sum([arr[i] for i in plurals])

def inflect_task(fname,model,method):
    inputs = []
    verbs = []
    attr = []
    for line in open(fname):
        (attractors,isPlural,subj_index,verb_index,sentence) = line.split('\t')
        sentence = parse_sentence(sentence)
        verb_index = int(verb_index)
        inputs.append(np.array(sentence))
        verbs.append(verb_index-1)
        attr.append(int(attractors))
    total = len(inputs)
    inputs = np.array(inputs)
    verbs = np.array(verbs)
    attr = np.array(attr)

    predictions = model.predict(inputs, batch_size = batch_size, verbose = 1)
    correct = {}
    total = {}
    for i in range(50):
        correct[i] = 0
        total[i] = 0
    if method == "inflection":
        for i in range(len(inputs)):
            w = inputs[i][verbs[i]]
            if isInflectibe(w):
                v = invertPlurality(w)
                total[attr[i]] += 1
                if predictions[i][verbs[i]][w] > predictions[i][verbs[i]][v]:
                    correct[attr[i]] += 1
    else:
        for i in range(len(inputs)):
            if isPluralPrediction(predictions[i][verbs[i]]) == isPluralWord(inputs[i][verbs[i]]):
                correct[attr[i]] += 1
            total[attr[i]] += 1
    print ("...")
    print ("|  attractors | correct | total | accuracy |")
    print ("|-")
    for k in total.keys():
        if total[k] > 0:
            print ("|", k, "|", correct[k], "|",total[k], "|", correct[k] / float (total[k]), "|")

language_model = None
while len(commands) > 0:
    cmd = pop()
    if cmd == 'load':
        filepath = pop()
        print ("Loading and compiling model: " + filepath)
        language_model = keras.models.load_model(filepath)
        print_model(language_model)
    elif cmd == 'create':
        language_model = Sequential([
            Embedding(n_vocab, embedding_size, input_length=MAXLEN),
            LSTM(lstm_memory_size, return_sequences=True, dropout_U=0.1, dropout_W=0.1, consume_less='gpu'),
            LSTM(lstm_memory_size, return_sequences=True, dropout_U=0.1, dropout_W=0.1, consume_less='gpu'),
            TimeDistributed(Dense(n_vocab, activation='softmax'))])
        language_model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        print_model(language_model)
    elif cmd == 'train':
         epochs = int(pop())
         language_model.fit_generator(generator = sequence_generator(corpus_files=train_set),
                                      validation_data = sequence_generator(corpus_files=val_set),
                                      nb_val_samples = batch_size * 100,
                                      samples_per_epoch=batch_size * 1000 * 2,
                                      nb_epoch=epochs,
                                      callbacks = [ModelCheckpoint(checkpoint_path)])

                                                                   # , save_best_only=True
         language_model.save(model_path)
    elif cmd == 'words':
        cmd = pop()
        if cmd == 'all':
            words = vocab.keys()
        elif cmd == 'verbs':
            words = plural_toks + singular_toks
        else:
            words = cmd.split()
        sentence = pop().split()
        predictions = language_model.predict(np.array([pad(encoder(sentence))]))[0]

        for j in range(len(sentence)):
            print ("---------" + str(j) + " ---- After: " + str(pad(sentence)[j]))
            out = [(predictions[j][vocab[wd]],wd) for wd in words]
            out.sort() # sorts "in-place", yay python T_T
            for pred,wd in out[-10:]:
                print (wd + " -> " + str(pred))
    elif cmd == 'sentence':
        sentence = pop().split()
        coded_sentence = encoder(sentence)
        print ("coded sentence = " + str(coded_sentence))
        predictions = language_model.predict(np.array([pad(coded_sentence)]))[0]
        score = 0
        for j in range(0,len(sentence)-1):
            probj = predictions[j][coded_sentence[j+1]]
            print (str(probj) + " = P("+ str(sentence[j+1]) + " | " + ' '.join(sentence[:j+1]) +")")
            score += math.log(probj)
        print ("Sentence score: " + str(score))
    elif cmd == 'inflect':
        testfilename = pop()
        inflect_task(testfilename,language_model, "inflection")
    elif cmd == 'predict':
        testfilename = pop()
        inflect_task(testfilename,language_model, "prediction")
    else:
        print ('Unknown command: ' + cmd)




