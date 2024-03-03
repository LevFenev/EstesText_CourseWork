import gensim
from gensim.models import Word2Vec
import pandas as pd
import re
import os
import numpy as np
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"

w2v_model = Word2Vec.load("resources/model_fenev.model")

dir_path = 'files/fenev'
files = os.listdir(dir_path)
means = {}

for i in list(os.listdir(dir_path)):
    results = 0
    words = 1
    with open(f"{dir_path}/{i}", 'r', encoding='utf-8') as file:
        print(f"{dir_path}/{i}")
        for line in file:
            line = re.sub(patterns, ' ', line)
            for word in line.split():
                if w2v_model.wv.has_index_for(word) and len(word) > 3:
                # if w2v_model.wv.has_index_for(word): для евы
                    results+=w2v_model.wv.get_vector(word).sum()
                    words+=1
    means[f"{dir_path}/{i}"] = results / words
w2v_model.wv.sort_by_descending_frequency()

r = []
for key in means:
    r.append(means[key])
r = sorted(r)

vectors = list(split(r,3))

vec_mean = []
for i in vectors:
    sum = 0
    count = 0
    for vec in i:
        sum+=vec
        count+=1
    vec_mean.append(sum/count)

for i in vec_mean:
    print(i)
    print(w2v_model.wv.similar_by_vector(np.array(i), topn=7, restrict_vocab=None))
    # print(w2v_model.wv.most_similar(positive=[np.array(i)], topn=7))

