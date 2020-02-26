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
# path_to_wiki_dump = datapath("/home/rohola/codes/topical_language_generation/wikipedia/enwiki-latest-pages-articles.xml.bz2")
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

import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt


# dist1 = torch.from_numpy(np.array([0.1, 0.4, 0.2, 0.3]))
# dist2 = torch.from_numpy(np.array([0.1, 0.4, 0.2, 0.3]))
# #dist2 = torch.from_numpy(np.array([0.7, 0.0, 0.1, 0.2]))
#
# logits1 = torch.log(dist1)
# logits2 = torch.log(dist2)
#
# #indices = logits2 == -float("Inf")
# indices = logits2 < -10e5
# logits2[indices] = logits1[indices]
#
# res1 = F.softmax(logits1)
# res2 = F.softmax((logits1+logits2)/2)
#
# plt.bar(np.arange(len(dist1)), res1)
# print(res1)
# plt.show()
#
# plt.bar(np.arange(len(dist1)), res2)
# print(res2)
# plt.show()


# probs = torch.Tensor([0.0, 0.5, 0.3, 0.2])
# scores = torch.Tensor([0.8, 0.0, 0.1, 0.1])
#
# probs[scores>0.3] = 1.0
# scores[scores==0]=1
#
# total_probs = F.softmax(torch.mul(probs, scores))
# print(total_probs)
#
# total_probs = F.softmax(torch.add(probs, scores))
# print(total_probs)

# import pandas as pd
# anes = pd.read_csv("/home/rohola/Downloads/ANES-elfardy-mdiab-ccb-dataset/*SEM-Split/ANES-AMT-*SEM-Test.csv", sep='\t')
#
# for index, row in anes.iterrows():
#     for col_name in anes.columns.values:
#         print(row[col_name])
#     print("##################")


from torch.distributions import kl_divergence
from torch.distributions.beta import Beta
from torch.distributions.categorical import Categorical


m = Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]))
print(m.sample())

print(m.entropy())

p = Beta(concentration0=1, concentration1=1)
q = Beta(concentration0=6, concentration1=2)
kldiv = kl_divergence(p, q)
print(kldiv)

# kl_scale = 1
# kl_loss = kl_scale * ((corrected_probs * (corrected_probs / unpert_probs).log()).sum())
