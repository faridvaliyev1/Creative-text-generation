import ujson as json
import pandas as pd
import numpy as np 
import nltk
from copy import deepcopy as dcopy
from collections import Counter
nltk.download('punkt')
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from random import shuffle

max_n_sent = 30
min_n_sent = 10

records = map(json.loads, open('gutenberg_poetry.ndjson'))
df = pd.DataFrame.from_records(records)
print(df)

df_1 = df[df.index > 25]


text_1 = '\n'.join(np.array(df_1)[:,0])
print('Total characters:', len(text_1))

text = text_1

print("Length of text: ", len(text))

text = text_1.lower()

text_copy = dcopy(text)

text = dcopy(text_copy)
cntr = Counter(nltk.word_tokenize(text))

most_words = list(cntr.most_common(8000-3))

print("Total words: ", len(list(cntr.most_common(10000000))))

print('The number of appearances of the least frequent word:', most_words[-1][0], most_words[-1][1])

vocab = dict()
for i in range(len(most_words)):
    vocab[most_words[i][0]] = i
vocab['']=len(vocab) 
vocab['_UNK_']=len(vocab) 
vocab['_START_']=len(vocab)

d = vocab
list_d = list(d.items())
list_d.sort(key=lambda i: i[1])

word = []
for i in range(len(list_d)):
    word.append(list_d[i][0])
print('Words in vocab:', len(word))

def merge_list(counter,my_list):
    iterator=0
    new_list=[]
    sentence=""
    for x in my_list:
        if iterator==counter:
            new_list.append(sentence)
            sentence=x
            iterator=0
        else:
            sentence+=x
        iterator+=1
    
    return new_list

text_processed = []
sets = text.split('\n')
sets = merge_list(4,sets)# -|-
for i, sent in enumerate(sets):
    if(i%(len(sets)//10)==0):
        print(float(i)/float(len(sets)), '%')
    text_processed.append(nltk.word_tokenize(sent))

min_len = len(text_processed[0])
max_len = len(text_processed[0])

sr_len = 0
for i in range(len(text_processed)):
  sr_len += len(text_processed[i])
  min_len = min(min_len, len(text_processed[i]))
  max_len = max(max_len, len(text_processed[i]))

print("Sentences: ", len(text_processed))
print("Minimal length", min_len)
print("Average length", sr_len/len(text_processed))
print("Max length", max_len)
print("Number of words", sr_len)

text_processed_copy = dcopy(text_processed)

text_processed = dcopy(text_processed_copy)

DEL_sentences=0
len_texts = len(text_processed)

i=0
ii=0
while i < len(text_processed):
  if(ii%(len_texts//100) == 0):
    print(ii//(len_texts//100), u'In Processing: ', len(text_processed), u"Deleted:", DEL_sentences, u'Saved: ', ii-DEL_sentences)
  
  if(len(text_processed[i]) < min_n_sent):# or len(text_processed[i]) > max_n_sent):#
    del(text_processed[i])
    DEL_sentences+=1
  else:
    text_processed[i] = text_processed[i][:max_n_sent]
    i+=1
  ii+=1
print('Number of Sentences: ', len(text_processed)) #112138 #214986 #571377


DEL_sentences=0
len_texts = len(text_processed)
i=0
ii=0
while i < len(text_processed):
  if(ii%(len_texts//100) == 0):
    print(ii//(len_texts//100), u'In Processing: ', len(text_processed), u"Deleted:", DEL_sentences, u'Saved: ', ii-DEL_sentences)
  
  text_processed[i] = text_processed[i][:max_n_sent]
  ln_n = len(text_processed[i])
  ln_s = 0
  j=0
  while j < ln_n:
    if(text_processed[i][j] not in word):
      
      text_processed[i][j] = '_UNK_'
      ln_s+=1
    j+=1
  if(ln_s>0):
    del(text_processed[i])
    DEL_sentences+=1
  else:
    i+=1
  ii+=1
print("Deleted number of sentences (with unknown words):", DEL_sentences)
print('Number of the sentences left: ', len(text_processed))

text_del_processed_copy = dcopy(text_processed)

text_processed = dcopy(text_del_processed_copy)

for i in range(len(text_processed)):
  for j in range(len(text_processed[i])):
    text_processed[i][j] = vocab[text_processed[i][j]]

max_n_sent = max_n_sent
for i in range(len(text_processed)):
  while len(text_processed[i]) < max_n_sent:
    text_processed[i].append(vocab[''])




obj = (word, vocab)
output = open(r'datasets\kaggle_poems\full_vocab_8000.pkl', 'wb')
pickle.dump(obj, output, 2)
output.close()

text_shuffle_copy = dcopy(text_processed)

text_processed = dcopy(text_shuffle_copy)

print('The number of sentences: ', len(text_processed)) #112138 #214986
for i in range(len(text_processed)):
  text_processed[i] = ' '.join(map(str, text_processed[i]))



shuffle(text_processed)

print('The number of samples: ', len(text_processed))


text_file = open(r"datasets\kaggle_poems\full_merge8000_train.txt", "w")
text_file.write('\n'.join(map(str, text_processed[:60000])) + '\n')
text_file.close()

text_file = open(r"datasets\kaggle_poems\full_merge8000_test.txt", "w")
text_file.write('\n'.join(map(str, text_processed[-6000:])) + '\n')
text_file.close()


