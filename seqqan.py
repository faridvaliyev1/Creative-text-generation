from __future__ import print_function
from math import ceil
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn

import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.init as init

import helpers
from Generator import Generator
from Discriminator import Discriminator

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from datetime import datetime
import nltk
import random
from scipy import stats

DATASET = 'merge'
CUDA = False

BATCH_SIZE = 25
MLE_TRAIN_EPOCHS = 30
DIS_TRAIN_ITERATIONS = 300
ADV_TRAIN_EPOCHS = 25
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 150
GEN_HIDDEN_DIM = 200
DIS_EMBEDDING_DIM = 150
DIS_HIDDEN_DIM = 150

START_LETTER = None
MAX_SEQ_LEN = None
VOCAB_SIZE = None
FILE_PATHS = None
CLOSING_WORD = None

if DATASET == 'merge': 
    START_LETTER = 0
    MAX_SEQ_LEN = 30
    VOCAB_SIZE = 8000
    CLOSING_WORD = 7999
    FILE_PATHS = {'train': r'datasets/kaggle_poems/full_merge8000_train.txt', 'test': r'datasets/kaggle_poems/full_merge8000_test.txt',
                'vocab': r'datasets/kaggle_poems/full_vocab_8000.pkl', 'saved_models': r'saved_models/kaggle_poems'}
if DATASET == 'full': 
    START_LETTER = 0
    MAX_SEQ_LEN = 10
    VOCAB_SIZE = 5000
    CLOSING_WORD = 4999
    FILE_PATHS = {'train': r'datasets/kaggle_poems/full_larger_train.txt', 'test': r'datasets/kaggle_poems/full_larger_test.txt',
                'vocab': r'datasets/kaggle_poems/full_vocab.pkl', 'saved_models': r'saved_models/kaggle_poems'}
if DATASET == 'love': 
    START_LETTER = 0
    MAX_SEQ_LEN = 10
    VOCAB_SIZE = 5000
    CLOSING_WORD = 4999
    FILE_PATHS = {'train': r'datasets/kaggle_poems/love_train.txt', 'test': r'datasets/kaggle_poems/love_test.txt',
                'vocab': r'datasets/kaggle_poems/love_vocab.pkl', 'saved_models': r'saved_models/kaggle_poems'}


torch.random.manual_seed(10)
np.random.seed(15)


def sampler_example(batch_size):
  x = data_file_train[np.random.randint(0, len(data_file_train), batch_size)]
  y = np.concatenate([x[:, 1:], np.zeros([batch_size, 1])+VOCAB_SIZE-2], axis=-1)
  return x, y


def sampler_example_test(batch_size):
  x = data_file_test[np.random.randint(0, len(data_file_test), batch_size)]
  y = np.concatenate([x[:, 1:], np.zeros([batch_size, 1])+VOCAB_SIZE-2], axis=-1)
  return x, y



def train_generator_MLE(gen, gen_opt, real_samples_train, real_samples_test, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        
        for i in range(0, len(real_samples_train), BATCH_SIZE):
            inp_train, target_train = helpers.prepare_generator_batch(real_samples_train[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp_train, target_train)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(len(real_samples_train) / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(len(real_samples_train) / float(BATCH_SIZE)) / MAX_SEQ_LEN
        print(' average_train_NLL = %.4f' % total_loss, end='')

       
        test_loss = 0

        for i in range(0, len(real_samples_test), BATCH_SIZE):
            inp_test, target_test = helpers.prepare_generator_batch(real_samples_test[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                              gpu=CUDA)
            loss = gen.batchNLLLoss(inp_test, target_test)
            test_loss += loss.data.item()
        test_loss = test_loss / ceil(len(real_samples_test) / float(BATCH_SIZE)) / MAX_SEQ_LEN
        print(' average_test_NLL = %.4f' % test_loss)

def test_mle(gen, real_samples_train, real_samples_test):
  test_loss = 0
  for i in range(0, len(real_samples_train), BATCH_SIZE):
      inp_test, target_test = helpers.prepare_generator_batch(real_samples_train[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                        gpu=CUDA)
      loss = gen.batchNLLLoss(inp_test, target_test)
      test_loss += loss.data.item()
  test_loss = test_loss / ceil(len(real_samples_train) / float(BATCH_SIZE)) / MAX_SEQ_LEN
  print('average_train_NLL = %.4f' % test_loss, end='')

  test_loss = 0
  for i in range(0, len(real_samples_test), BATCH_SIZE):
      inp_test, target_test = helpers.prepare_generator_batch(real_samples_test[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                        gpu=CUDA)
      loss = gen.batchNLLLoss(inp_test, target_test)
      test_loss += loss.data.item()
  test_loss = test_loss / ceil(len(real_samples_test) / float(BATCH_SIZE)) / MAX_SEQ_LEN
  print(' average_test_NLL = %.4f' % test_loss)


def train_generator_PG(gen, gen_opt, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*4)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()
    print()



def train_discriminator(discriminator, dis_opt, real_data_samples, generator, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training
    pos_val = real_data_samples[np.random.randint(0, len(real_data_samples), 500)] #sampler_example(250)
    neg_val = generator.sample(500)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        val_pred = discriminator.batchClassify(val_inp)
        print('Before Training: val_acc = %.4f' % (
            torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/1000.))
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/1000.))



def BLEU(reference_sample, test_sample, print_iteration=100, flag_print=True):
  if flag_print:
    print("--- --- ---\nStart BLEU")
  pad = CLOSING_WORD
  #################################################
  reference = []
  for line in reference_sample:
    candidate = []
    for i in line:
      if i == pad:
        break
      candidate.append(i)

    reference.append(candidate)
  #################################################
  hypothesis_list_leakgan = []
  for line in test_sample:
    while line[-1] == str(pad):
      line.remove(str(pad))
    hypothesis_list_leakgan.append(line)
  #################################################
  random.shuffle(hypothesis_list_leakgan)
  #################################################

  smoothing_function = SmoothingFunction().method1

  mass_bleu = []
  for ngram in range(2,6):
      weight = tuple((1. / ngram for _ in range(ngram)))
      bleu_leakgan = []
      bleu_supervise = []
      bleu_base2 = []
      num = 0
      for h in hypothesis_list_leakgan:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, h, weight, smoothing_function = smoothing_function)
          num += 1
          bleu_leakgan.append(BLEUscore)

          if num%print_iteration == 0 and flag_print:
            print(ngram, num, sum(bleu_leakgan)/len(bleu_leakgan))
          
      mass_bleu.append(1.0 * sum(bleu_leakgan) / len(bleu_leakgan))
      if flag_print:
        print('--- --- ---')
        print(len(weight), '-gram BLEU score : ', 1.0 * sum(bleu_leakgan) / len(bleu_leakgan), "\n")
  return mass_bleu



def save_models(data_file_tensor_train, gen, dis, gen_optimizer, dis_optimizer, name):
  state = {
      'default_parameters': {'VOCAB_SIZE': VOCAB_SIZE, 'MAX_SEQ_LEN': MAX_SEQ_LEN, 'GEN_EMBEDDING_DIM': GEN_EMBEDDING_DIM,
                             'GEN_HIDDEN_DIM': GEN_HIDDEN_DIM, 'DIS_EMBEDDING_DIM': DIS_EMBEDDING_DIM, 'DIS_HIDDEN_DIM': DIS_HIDDEN_DIM},
      'data_file_tensor_train': data_file_tensor_train,
      'gen_state_dict': gen.state_dict(),
      'dis_state_dict': dis.state_dict(),
      'gen_optimizer': gen_optimizer.state_dict(),
      'dis_optimizer': dis_optimizer.state_dict(),
  }
  torch.save(state, name)



def load_models(name):
  if CUDA:
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  print('state')
  state = torch.load(name, map_location=device)

  print('default_parameters')
  VOCAB_SIZE = state['default_parameters']['VOCAB_SIZE']
  MAX_SEQ_LEN = state['default_parameters']['MAX_SEQ_LEN']
  GEN_EMBEDDING_DIM = state['default_parameters']['GEN_EMBEDDING_DIM']
  GEN_HIDDEN_DIM = state['default_parameters']['GEN_HIDDEN_DIM']
  DIS_EMBEDDING_DIM = state['default_parameters']['DIS_EMBEDDING_DIM']
  DIS_HIDDEN_DIM = state['default_parameters']['DIS_HIDDEN_DIM']

  print('data_file_tensor_train')
  data_file_tensor_train = torch.tensor(state['data_file_tensor_train'])

  print('Generator')
  gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
  gen.load_state_dict(state['gen_state_dict'])
  gen_optimizer = optim.Adam(gen.parameters(), lr=0.001)
  gen_optimizer.load_state_dict(state['gen_optimizer'])

  print('Discriminator')
  dis = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
  dis.load_state_dict(state['dis_state_dict'])
  dis_optimizer = optim.Adagrad(dis.parameters())
  dis_optimizer.load_state_dict(state['dis_optimizer'])

  print('CUDA')
  if CUDA:
    data_file_tensor_train = data_file_tensor_train.cuda()
    gen = gen.cuda()
    dis = dis.cuda()
  
  return [data_file_tensor_train, gen, dis, gen_optimizer, dis_optimizer,
          VOCAB_SIZE, MAX_SEQ_LEN, GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM]



import pickle

vocab_file = FILE_PATHS['vocab']
word, vocab = pickle.load(open(vocab_file, 'rb'))

f = open(FILE_PATHS['train'], 'r')
data_file_train = []
for line in f:
  line = line.replace('\n', '')
  line = line.split()
  for i in range(len(line)):
    line[i] = int(line[i])
  data_file_train.append(line)
data_file_train = np.array(data_file_train)[:, :MAX_SEQ_LEN]
print("Examples in training set: ", len(data_file_train))

f = open(FILE_PATHS['test'], 'r')
data_file_test = []
for line in f:
  line = line.replace('\n', '')
  line = line.split()
  for i in range(len(line)):
    line[i] = int(line[i])
  data_file_test.append(line)
data_file_test = np.array(data_file_test)[:, :MAX_SEQ_LEN]
print("Examples in test set: ", len(data_file_test))


print("samples from training set")
samples = sampler_example_test(50)[0]
output_function = []
for samp in samples:
  line = [word[x] for x in samp]
  line = ' '.join(line)
  output_function.append(line)

for i, output in enumerate(output_function):
  print("#", i, "\tSample: ", output)    



print("BLEU score of training set")
BLEU(data_file_test.tolist(), data_file_train[:500].tolist(), print_iteration=100)



gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
dis = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

if CUDA:
  gen = gen.cuda()
  dis = dis.cuda()
  data_file_tensor_train = torch.tensor(data_file_train).cuda()
  data_file_tensor_test = torch.tensor(data_file_test).cuda()
else:
  data_file_tensor_train = torch.tensor(data_file_train)
  data_file_tensor_test = torch.tensor(data_file_test)

gen_optimizer = optim.Adam(gen.parameters(), lr=0.001) #, lr=0.001
dis_optimizer = optim.Adagrad(dis.parameters())


test_mle(gen, data_file_tensor_train, data_file_tensor_test)

print('Training of generator with MLE objective...')
gen_optimizer = optim.Adam(gen.parameters())#, lr=0.0002
train_generator_MLE(gen, gen_optimizer, data_file_tensor_train, data_file_tensor_test, 3) # MLE_TRAIN_EPOCHS

test_mle(gen, data_file_tensor_train, data_file_tensor_test)

save_models(data_file_tensor_train, gen, dis, gen_optimizer, dis_optimizer,
            FILE_PATHS['saved_models'] + r'/' + r'seqgan_mle.pytorch')


test_mle(gen, data_file_tensor_train, data_file_tensor_test)


print("Examples of text, generated with MLE objective")
degree = 2
print("Degree:", degree)
samples = gen.sample(60, degree=degree).cpu().detach().numpy()

output_function = []
for i, samp in enumerate(samples):
  line = [word[x] for x in samp]
  line = ' '.join(line)
  output_function.append(line)
  bleu = BLEU(data_file_test.tolist(), [samp], flag_print=False)
  print("#", i, "\tExample: ", line)



from tqdm import tqdm
samples1 = gen.sample(50, degree=1.5).cpu().detach().numpy().tolist()
samples2 = gen.sample(50, degree=1.5).cpu().detach().numpy().tolist()

from sklearn.metrics import jaccard_score
diversity = []
for sample1 in tqdm(samples1):
    scores = []
    for sample2 in samples2:
        scores.append(jaccard_score(sample1, sample2, average = 'macro'))
    total = np.max(scores)
    diversity.append(total)

print(1-np.mean(diversity))




print('Pre-training discriminator...')
# dis_optimizer = optim.Adagrad(dis.parameters()) # , lr=0.0001
# dis_optimizer = optim.Adam(gen.parameters(), lr=0.001, weight_decay=1e-5)#0.001
# dis_optimizer = optim.Adadelta(gen.parameters())
# dis_optimizer = optim.Adagrad(dis.parameters())#, lr=0.0001)#, weight_decay=1e-5)
#, weight_decay=1e-5) 
train_discriminator(dis, dis_optimizer, data_file_tensor_train, gen, 10, 1)#25, 1 | (15, 3), (25, 1)


save_models(data_file_tensor_train, gen, dis, gen_optimizer, dis_optimizer,
            FILE_PATHS['saved_models'] + r'/' + r'seqgan_pretraining_dis.pytorch')



test_mle(gen, data_file_tensor_train, data_file_tensor_test)





print('\nStarting Adversarial Training...')

for epoch in range(ADV_TRAIN_EPOCHS):# ADV_TRAIN_EPOCHS
    print('\n--------\nEPOCH %d\n--------' % (epoch+1))
    
    print('\nAdversarial Training Generator : ', end='')
    train_generator_PG(gen, gen_optimizer, dis, 1)#

    
    print('\nTesting Generator : ', end='')
    test_mle(gen, data_file_tensor_train, data_file_tensor_test)

    
    print('\nAdversarial Trainin+++++++++++++++++++++++++g Discriminator : ')
    train_discriminator(dis, dis_optimizer, data_file_tensor_train, gen, 3, 1)#3, 1


save_models(data_file_tensor_train, gen, dis, gen_optimizer, dis_optimizer,
            FILE_PATHS['saved_models'] + r'/' + r'seqgan_adversarial_training.pytorch')


test_mle(gen, data_file_tensor_train, data_file_tensor_test)

print("Examples, generated after SeqGAN training")
degree = 2
print("Degree:", degree)
samples = gen.sample(10, degree=degree,start_letter = 0).cpu().detach().numpy()
output_function = []
for i, samp in enumerate(samples):
  line = [word[x] for x in samp]
  line = ' '.join(line)
  output_function.append(line)
  bleu = BLEU(data_file_test.tolist(), [samp], flag_print=False)
  print("#", i, "\tExample: ", line)


print("BLEU score of text, generated after SeqGAN training")
degree = 1
print("Degree:", degree)

BLEU(data_file_test.tolist(), gen.sample(500, degree=degree).cpu().detach().numpy().tolist(), print_iteration=100)


print("Examples of generated sentences with SeqGAN")
degree = 1.5
print("Degree:", degree)
samples = gen.sample(20, degree=degree).cpu().detach().numpy()

output_function = []
for i, samp in enumerate(samples):
  line = [word[x] for x in samp]
  line = ' '.join(line)
  output_function.append(line)
  bleu = BLEU(data_file_test.tolist(), [samp], flag_print=False)
  print("#", i, "\tExample: ", line)




BLEU(data_file_test.tolist(), gen.sample(500, degree=degree).cpu().detach().numpy().tolist(), print_iteration=100)


[data_file_tensor_train, gen, dis, gen_optimizer, dis_optimizer,
VOCAB_SIZE, MAX_SEQ_LEN, GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, DIS_EMBEDDING_DIM,
 DIS_HIDDEN_DIM] = load_models(FILE_PATHS['saved_models'] + r'/' + r'seqgan_adversarial_training.pytorch')#[seqgan_mle, seqgan_pretraining_dis, seqgan_adversarial_training]

if(CUDA):
  gen = gen.cuda()
  dis = dis.cuda()
  data_file_tensor_train = torch.tensor(data_file_tensor_train).cuda()
  data_file_tensor_test = torch.tensor(data_file_test).cuda()



degree=1.5



samples = gen.sample(150, degree=degree).cpu().detach().numpy().tolist()
# samples = data_file_test.tolist()[:500]
train_samples = data_file_train.tolist()
n = 0
for i in range(len(samples)):
  if samples[i] in train_samples:
    n += 1
  if i%(len(samples)//10) == 0:
    print(i/len(samples)*100, "%")
print("Number of total overlaps with training dataset: ", n, "из", len(samples))
print("Plagiarism: ", n/len(samples)*100, "%")


N_samples = 1500
samples_1 = gen.sample(N_samples, degree=degree).cpu().detach().numpy().tolist()
samples_2 = gen.sample(N_samples, degree=degree).cpu().detach().numpy().tolist()

n = 0
for i in range(len(samples_1)):
  if samples_1[i] in samples_2:
    n += 1
  if i%(len(samples_1)//10) == 0:
    print(i/len(samples_1)*100, "%")

print("Original generated sentences")
print("The number of overlaps with first batch and second batch: ", n, len(samples_1))
print("Plagiat: ", n/len(samples_1)*100, "%")
print("Originality: ", (1-n/len(samples_1))*100, "%")





from tqdm import tqdm
samples = gen.sample(50, degree=1.5).cpu().detach().numpy().tolist()
from sklearn.metrics import jaccard_score
novelty = []
for sample in tqdm(samples):
    scores = []
    for training_sent in train_samples:
        scores.append(jaccard_score(sample, training_sent, average = 'macro'))
    total = np.max(scores)
    novelty.append(total)

print(1-np.mean(novelty))



from tqdm import tqdm
samples1 = gen.sample(50, degree=1.5).cpu().detach().numpy().tolist()
samples2 = gen.sample(50, degree=1.5).cpu().detach().numpy().tolist()

from sklearn.metrics import jaccard_score
diversity = []
for sample1 in tqdm(samples1):
    scores = []
    for sample2 in samples2:
        scores.append(jaccard_score(sample1, sample2, average = 'macro'))
    total = np.max(scores)
    diversity.append(total)
    
print(1-np.mean(diversity))



print("Examples of generated sentences with SeqGAN")
degree = 1
print("Degree:", degree)
samples = gen.sample(50, degree=degree).cpu().detach().numpy()

output_function = []
for i, samp in enumerate(samples):
  line = [word[x] for x in samp]
  line = ' '.join(line)
  output_function.append(line)
  bleu = BLEU(data_file_test.tolist(), [samp], flag_print=False)
  print("#", i, "\tExample: ", line)





print("Examples of generated sentences with SeqGAN")
degree = 1.5
print("Degree:", degree)
samples = gen.sample(50, degree=degree).cpu().detach().numpy()

output_function = []
for i, samp in enumerate(samples):
  line = [word[x] for x in samp]
  line = ' '.join(line)
  output_function.append(line)
  bleu = BLEU(data_file_test.tolist(), [samp], flag_print=False)
  print("#", i, "\tExample: ", line)




print("Examples of generated sentences with SeqGAN")
degree = 2
print("Degree:", degree)
samples = gen.sample(50, degree=degree).cpu().detach().numpy()

output_function = []
for i, samp in enumerate(samples):
  line = [word[x] for x in samp]
  line = ' '.join(line)
  output_function.append(line)
  bleu = BLEU(data_file_test.tolist(), [samp], flag_print=False)
  print("#", i, "\tExample: ", line)


