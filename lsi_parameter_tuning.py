from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel, CoherenceModel

from datasets.topical_dataset import TopicalDataset
from topical_tokenizers import TransformerGPT2Tokenizer
from configs import LSIConfig
import wandb
import logging

logging.getLogger().setLevel(logging.ERROR)

config_file = "configs/alexa_lsi_config.json"
#config_file = "configs/nytimes_lsi_config.json"
#config_file = "configs/anes_lsi_config.json"
#config_file = "configs/newsgroup_lsi_config.json"

config = LSIConfig.from_json_file(config_file)
# wandb.init(config=config, project="topical_language_generation_sweeps")

#data preparation
cached_dir = "/home/rohola/cached_models"
tokenizer = TransformerGPT2Tokenizer(cached_dir)
dataset = TopicalDataset(config.dataset_dir, tokenizer)

docs = [doc for doc in dataset]

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=config.no_below,
                           no_above=config.no_above)


corpus = [dictionary.doc2bow(doc) for doc in docs]
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi_model = LsiModel(corpus_tfidf,
                     id2word=dictionary,
                     num_topics=config.num_topics,
                     )

#cm = CoherenceModel(model=lsi_model, corpus=corpus, coherence='u_mass')
cm = CoherenceModel(model=lsi_model, texts=docs, dictionary=dictionary, coherence='c_w2v')
# coherence = cm.get_coherence()
# print("coherence: ", coherence)
#wandb.log({"coherence": cm.get_coherence()})



