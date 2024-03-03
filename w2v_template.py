import gensim
from gensim.models import Word2Vec
import pandas as pd
import re

patterns = "[A-Za-zА-ЯЁа-яё\d!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"

response = []
# Import train_rel_2.tsv into Python
with open('resources/all_articles.tsv', encoding="utf-8") as f:
    lines = f.readlines()
    # print(lines)
    columns = lines[0].split('\t')
    for line in lines[1:]:
        temp = line.split('\t')
        # if temp[1] == '2':   # Select the Essay Set 2
        response.append(re.sub(patterns, ' ', temp[0]))  # Select "EssayText" as a corpus

data = pd.DataFrame(list(zip(response)))
data.columns = ['response']
response_base = data.response.apply(gensim.utils.simple_preprocess)

model = Word2Vec(
    sentences=response_base,
    min_count=30,
    window=2,
    vector_size=64,
    alpha=0.03,
    negative=10,
    min_alpha=0.0007,
    sample=6e-5
)

print(model.wv['анализ'])
# print(model.corpus_count)

# Train the model
model.build_vocab(response_base, update=True)
model.train(response_base, total_examples=model.corpus_count, epochs=model.epochs)
print(model.wv['анализ'])

model.save("resources/model_fenev.model")

# print(model.corpus_count)
print(model.wv.most_similar_to_given("поэзия", ["анализ", "разбор"]))
