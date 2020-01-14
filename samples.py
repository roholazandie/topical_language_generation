# model_name_or_path = "gpt2" #50257 tokens
# tokenizer_class = GPT2Tokenizer
# tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
#
# doc = "this is here. here is good."
# tokens = tokenizer.tokenize(doc)
# print(tokens)
#
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
#
# ids = tokenizer.encode(doc, add_special_tokens=False)
# print(ids)


# from gensim.test.utils import datapath
# from gensim.corpora import WikiCorpus
#
# path_to_wiki_dump = datapath("/home/rohola/codes/topic_modeling_language_generation/wikipedia/enwiki-latest-pages-articles.xml.bz2")
#
# for vec in WikiCorpus(path_to_wiki_dump):
#     print(vec)

# from collections import defaultdict
# docids = [[3, 5, 2, 3], [2,3, 3], [1,2,3]]
# dfs = defaultdict(int)
# for docid in docids:
#     for id in docid:
#         dfs[id]+=1
#
# print(dict(dfs))
# import pickle
#
# mytok2freq = pickle.load(open("mytoken2frequency.p", 'rb'))
# tok2freq = pickle.load(open("token2freq.p", 'rb'))
#
# shared_items = {k: mytok2freq[k] for k in mytok2freq if k in tok2freq and mytok2freq[k] == tok2freq[k]}

# from lda_model import LDAModel
#
# lda_config_file = "configs/alexa_lda_config.json"
# lda_model = LDAModel(lda_config_file)
# theta = lda_model.get_theta_matrix()
# psi = lda_model.get_psi_matrix()
#
# print(psi)
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt

# scores = [0.001, 0.01, 0.2, 0.9, 0.3, 0.2, 0.002, 0.004, 0.03]
# p1 = softmax(scores)
# y_pos = np.arange(len(scores))
# plt.bar(y_pos, p1)
# plt.show()
#
# k = np.power(scores, 2)
# print(k)
# p2 = softmax(k)
# plt.bar(y_pos, p2)
#
# plt.show()
#
#
# n = [-2,-3, -np.inf]
#
# a1 = softmax(n)
# a2 = softmax(np.multiply(n, 2))
# print(a1)
# print(a2)


import multiprocessing as mp

class Foo():
    def ss(self):
        pass

    @staticmethod
    def work(self):
        pass

if __name__ == '__main__':
    pool = mp.Pool()
    foo = Foo()
    pool.apply_async(foo.work)
    pool.close()
    pool.join()