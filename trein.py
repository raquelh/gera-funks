from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


import numpy as np
import os
import time
import pandas as pd


df = pd.read_json('funks.json')

#df2 = df.letter

#df2.to_csv('funkemtxt.txt',index=None, header=None)

text = open('funkemtxt.txt', 'rb').read().decode(encoding='utf-8')
print ('Length of text: {} characters'.format(len(text)))


vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

#Process text


# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

#Create training examples and targets¶
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])
