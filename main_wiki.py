from topical_tokenizers import GPT2Tokenizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.utils import ClippedCorpus
import numpy as np
from datasets.wikidata import WikiData
import pickle

# cached_dir = "/home/rohola/cached_models"
# model_name_or_path = "gpt2" #50257 tokens
# tokenizer_class = GPT2Tokenizer
# tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cached_dir)
#
# dirname = "/media/rohola/data/clean_wiki_text_json/"
# wiki = WikiData(dirname)
#
# doc_stream = (tokens for tokens in wiki)
# id2word_wiki = Dictionary(doc_stream)
# id2word = id2word_wiki.filter_extremes(no_below=20, no_above=0.1)

dirname = "/media/rohola/data/clean_wiki_text_json"
wikidata = WikiData(dirname)

doc_stream = (tokens for tokens in wikidata)
id2word_wiki = Dictionary(doc_stream)
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)



mm_corpus = MmCorpus('/home/rohola/codes/topical_language_generation/caches/wiki_bow.mm')
clipped_corpus = ClippedCorpus(mm_corpus, 4000)


######################################
# Set training parameters.
num_topics = 20
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

model = LdaModel(
    corpus=wiki,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(wiki)


np.save("wiki_topics.npy", model.get_topics())
# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

pickle.dump(top_topics, open("top_topics.p", "wb"))