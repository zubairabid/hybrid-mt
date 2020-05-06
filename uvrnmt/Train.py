import math
import time
import spacy
import torch
import torchtext
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Transformer
from imagesfromsentence import get_features
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torchtext.data import Field, RawField, BucketIterator, TabularDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type=="cuda":
    print("Number of GPUs: ", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

#Open files
# europarl_en = open('./wmt14_en_fr/train.en', encoding='utf-8').read()[:100].split('\n')
# europarl_fr = open('./wmt14_en_fr/train.fr', encoding='utf-8').read()[:100].split('\n')

with open('./wmt14_en_fr/train.en', 'r') as f:
    europarl_en = f.read().split('\n')
with open('./wmt14_en_fr/train.fr', 'r') as f:
    europarl_fr = f.read().split('\n')

test_en = europarl_en[-100:]
test_fr = europarl_fr[-100:]

europarl_en = europarl_en[:5000]
europarl_fr = europarl_fr[:5000]
#Build tokenizers for both language using spacy
en = spacy.load('en')
fr = spacy.load('fr')

#Tokenizer functions
def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

def tokenize_fr(sentence):
    return [tok.text for tok in fr.tokenizer(sentence)]

#Defines a datatype together with instructions for converting to Tensor
EN_TEXT = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
FR_TEXT = Field(tokenize=tokenize_fr, init_token='<sos>', eos_token='<eos>', lower=True)
ORIG_ENG = RawField()
#Convert the data into CSV Format
#Tabular data works really well when we use Torchtext
raw_data = {'Original':[line for line in europarl_en], 'English':[line for line in europarl_en], 'French':[line for line in europarl_fr]}
df = pd.DataFrame(raw_data, columns=["Original","English","French"])

#Omit lengthy sentences and sentences whose length difference are not roughly close to each other
df['eng_len'] = df['English'].str.count(' ')
df['fr_len'] = df['French'].str.count(' ')
df = df.query('fr_len < 151 & eng_len < 151')
df = df.query('fr_len < eng_len * 1.5 & fr_len * 1.5 > eng_len')

#create train and validation set 
train, val = train_test_split(df, test_size=0.1)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)



# associate the text in the 'English' column with the EN_TEXT field, # and 'French' with FR_TEXT
data_fields = [('Original', ORIG_ENG), ('English', EN_TEXT), ('French', FR_TEXT)]
train,val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)

#Length of data sets
train_len = len(train)
val_len = len(val)


#Build Vocabulary to Index mapping
FR_TEXT.build_vocab(train, val)
EN_TEXT.build_vocab(train, val)

BATCHSIZE = 40

#Form iterator
train_iter = BucketIterator(train, batch_size=BATCHSIZE, sort_key=lambda x: len(x.French),
device=device, shuffle=True)
val_iter = BucketIterator(val, batch_size =BATCHSIZE, sort_key=lambda x: len(x.French),device=device, shuffle=True)


d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads, device)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    gpus = list(range(torch.cuda.device_count()))
    model = nn.DataParallel(model, device_ids=gpus)
model = model.to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

#Optimization inits and params 
lr = 5.0 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch_no, batch in enumerate(train_iter):
        source, target = batch.English.t(), batch.French.t()
        # the French sentence we input has all words except
        # the last, as it is using each word to predict the next
        batch_image_features =  get_features(batch.Original, 5)
        batch_image_features = torch.FloatTensor(batch_image_features)
        trg_input = target[:, :-1]
        # the words we are trying to predict
        targets = target[:, 1:].contiguous().view(-1)

        #Creates mask with 0s wherever there is padding in the input
        input_pad = EN_TEXT.vocab.stoi['<pad>']
        src_mask = (source != input_pad).unsqueeze(-2)
        
        #Creates same type of mask for target
        target_pad = FR_TEXT.vocab.stoi['<pad>']
        target_mask = (trg_input != target_pad).unsqueeze(-2)

        #Masking to ensure decoder user sequences up until that time stop
        size = trg_input.size(1)
        nopeak_mask = np.triu(np.ones((1,size,size)), k =1).astype('uint8')
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask)==0)
        #Final target mask
        nopeak_mask = nopeak_mask.to(device)
        target_mask = target_mask & nopeak_mask

        optimizer.zero_grad()
        
        source = source.to(device)
        trg_input = trg_input.to(device)
        batch_image_features = batch_image_features.to(device)
        src_mask = src_mask.to(device)
        target_mask = target_mask.to(device)

        output = model(source, trg_input, batch_image_features, src_mask, target_mask)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), targets, ignore_index=target_pad)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1
        if batch_no % log_interval == 0:# and batch_no > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'train loss {:5.2f} | train ppl {:8.2f}'.format(
                    epoch, batch_no, (train_len // BATCHSIZE), lr,
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for batch_no,batch in enumerate(val_iter):
            source, target = batch.English.t(), batch.French.t()
            batch_image_features =  get_features(batch.Original, 5)
            batch_image_features = torch.FloatTensor(batch_image_features)
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next    
            trg_input = target[:, :-1]
            # the words we are trying to predict
            targets = target[:, 1:].contiguous().view(-1)

            #Creates mask with 0s wherever there is padding in the input
            input_pad = EN_TEXT.vocab.stoi['<pad>']
            src_mask = (source != input_pad).unsqueeze(-2)
        
            #Creates same type of mask for target
            target_pad = FR_TEXT.vocab.stoi['<pad>']
            target_mask = (trg_input != target_pad).unsqueeze(-2)

            #Masking to ensure decoder user sequences up until that time stop
            size = trg_input.size(1)
            nopeak_mask = np.triu(np.ones((1,size,size)), k =1).astype('uint8')
            nopeak_mask = Variable(torch.from_numpy(nopeak_mask)==0)
            #Final target mask
            nopeak_mask = nopeak_mask.to(device)
            target_mask = target_mask & nopeak_mask

            source = source.to(device)
            trg_input = trg_input.to(device)
            batch_image_features = batch_image_features.to(device)
            src_mask = src_mask.to(device)
            target_mask = target_mask.to(device)

            output = eval_model(source, trg_input, batch_image_features, src_mask, target_mask)
            output_flat = output.view(-1, output.size(-1))
            loss = F.cross_entropy(output_flat, targets, ignore_index=target_pad)
            total_loss += loss.item()
    
    return total_loss# / (val_len - 1)

def translate(model, src, max_len = 80, custom_string=False):

    model.eval()
    input_pad = EN_TEXT.vocab.stoi['<pad>']
    if custom_string == True:
        src = tokenize_en(src)
        sentence=Variable(torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok in src]])).cuda()
        src_mask = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])
    for i in range(1, max_len):

        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()

        out = model.out(model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break
    return ' '.join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])


import pickle
import os

if os.path.isfile('model.pkl'):
    train = input("A trained model has been found. Train again? y/n")

if train == 'y':
    best_val_loss = float("inf")
    epochs = 50 # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

pred_fr = []
for sent in test_en:
    pred_fr.append(translate(model, sent, max_len=300, custom_string=True))


import nltk
BLEU_scores = []
for index in tqdm(range(len(test_fr))):
    BLEU_scores.append(
        nltk.translate.bleu_score.sentence_bleu([test_fr[index]], pred_fr[index], smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
print("Average BLEU Score:", np.mean(BLEU_scores))
