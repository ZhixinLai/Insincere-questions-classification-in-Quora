import os
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers


embed_size = 1200
max_features = 95000 
maxlen = 70 


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x
   
def split_text(x):
    x = wordninja.split(x)
    return '-'.join(x)

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df["question_text"] = train_df["question_text"].str.lower()
test_df["question_text"] = test_df["question_text"].str.lower()
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
train_df["question_text"]= train_df["question_text"].fillna("_##_").values
test_df["question_text"] = test_df["question_text"].fillna("_##_").values
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
train_df, val_df = train_test_split(train_df, test_size=0.01, random_state=2019) 
train_X = train_df["question_text"]
val_X = val_df["question_text"]
test_X = test_df["question_text"]    

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)
## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values  
#shuffling the data
np.random.seed(2019)
trn_idx = np.random.permutation(len(train_X))
val_idx = np.random.permutation(len(val_X))
train_X = train_X[trn_idx]
val_X = val_X[val_idx]
train_y = train_y[trn_idx]
val_y = val_y[val_idx]    

def check_the_word(train,test,embeddings_index):
    new_embeddings_index={}
    for i in tqdm(train):
        a=i.split()
        for j in a:
            if j in embeddings_index:
                new_embeddings_index[j]=embeddings_index[j]
    print(len(new_embeddings_index))
    for i in tqdm(test):
        a=i.split()
        for j in a:
            if j in embeddings_index:
                new_embeddings_index[j]=embeddings_index[j]
    print("new_embeddings_size:  ",len(new_embeddings_index))   
    return new_embeddings_index

def wiki():
    embeddings_index = {}
    f = open('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    new_embeddings_index=check_the_word(train_df["question_text"],test_df["question_text"],embeddings_index)
    return new_embeddings_index
embeddings_wiki=wiki()

from gensim.models import KeyedVectors
def google():
    EMBEDDING_FILE = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
    embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    new_embeddings_index=check_the_word(train_df["question_text"],test_df["question_text"],embeddings_index)
    return new_embeddings_index
embeddings_google=google()

def glove():
    embeddings_index = {}
    f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    new_embeddings_index=check_the_word(train_df["question_text"],test_df["question_text"],embeddings_index)
    return new_embeddings_index
embeddings_glove=glove()

def para():
    embeddings_index = {}
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    for line in tqdm(open(EMBEDDING_FILE, encoding="utf8", errors='ignore')):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    new_embeddings_index=check_the_word(train_df["question_text"],test_df["question_text"],embeddings_index)
    return new_embeddings_index
embeddings_para=para()

def mix():  
    
    empty_emb = np.zeros(300)
    embeddings_mix_index={}
    
    for i in tqdm(train_df["question_text"]):
        a=i.split()
        for j in a:
            embeddings_mix_glove=embeddings_glove.get(j,empty_emb)
            embeddings_mix_wiki=embeddings_wiki.get(j,empty_emb)
            embeddings_mix_para=embeddings_para.get(j,empty_emb)
            embeddings_mix_google=embeddings_google.get(j,empty_emb)
            embeddings_mix_index[j]=np.concatenate((embeddings_mix_glove,embeddings_mix_wiki,embeddings_mix_para,embeddings_mix_google),axis=0)
            
    for i in tqdm(test_df["question_text"]):
        a=i.split()
        for j in a:
            embeddings_mix_glove=embeddings_glove.get(j,empty_emb)
            embeddings_mix_wiki=embeddings_wiki.get(j,empty_emb)
            embeddings_mix_para=embeddings_para.get(j,empty_emb)
            embeddings_mix_google=embeddings_google.get(j,empty_emb)
            embeddings_mix_index[j]=np.concatenate((embeddings_mix_glove,embeddings_mix_wiki,embeddings_mix_para,embeddings_mix_google),axis=0)
            
    return embeddings_mix_index

embeddings_mix1=mix()

del  embeddings_glove
del  embeddings_para
del  embeddings_google
del  embeddings_wiki

def embed(embeddings_index):
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding1=embed(embeddings_mix1)

def model_cnn(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='elu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x) # New
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model    

def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_gru_atten_3(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train_pred(model):
    model.fit(train_X, train_y, batch_size=512,epochs=2, validation_data=(val_X, val_y))
    pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y

cnn_val_y,cnn_test_y=train_pred(model_cnn(embedding1))
lstm_val_y,lstm_test_y=train_pred(model_lstm_atten(embedding1))
gru_srk_atten_val_y,gru_srk_atten_test_y=train_pred(model_gru_srk_atten(embedding1))
lstm_du_val_y,lstm_du_test_y=train_pred(model_lstm_du(embedding1))
gru_atten_val_y,gru_atten_test_y=train_pred(model_gru_atten_3(embedding1))

blend_valid=np.concatenate((cnn_val_y,lstm_val_y,gru_srk_atten_val_y,lstm_du_val_y,gru_atten_val_y),axis=1)
blend_test=np.concatenate((cnn_test_y,lstm_test_y,gru_srk_atten_test_y,lstm_du_test_y,gru_atten_test_y),axis=1)

from numpy.linalg import inv
label=val_y.reshape(len(val_y),1)
beta=np.dot(inv(np.dot(blend_valid.T,blend_valid)),np.dot(blend_valid.T,label))

valid_predict=np.dot(blend_valid,beta)
test_predict=np.dot(blend_test,beta)

score=[]
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score+=[metrics.f1_score(val_y, (valid_predict>thresh).astype(int))]
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (valid_predict>thresh).astype(int))))
index=score.index(max(score))    
best=np.arange(0.1, 0.501, 0.01)[index]
print ("best thrshold is : ",best, " score : ", max(score))    

pred_test_y = (test_predict > best).astype(int)
print(np.sum(pred_test_y))
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)

