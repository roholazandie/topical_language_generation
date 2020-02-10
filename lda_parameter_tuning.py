from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel, CoherenceModel, LdaModel

from configs import LDAConfig
from datasets.topical_dataset import TopicalDataset
import wandb
import logging

logging.getLogger().setLevel(logging.ERROR)

#config_file = "configs/alexa_lsi_config.json"
#config_file = "configs/nytimes_lsi_config.json"
#config_file = "configs/anes_lsi_config.json"
from topical_tokenizers import TransformerGPT2Tokenizer



config_file = "configs/alexa_lda_config.json"

config = LDAConfig.from_json_file(config_file)
wandb.init(config=config, project="topical_language_generation_sweeps")

#data preparation
cached_dir = "/home/rohola/cached_models"
tokenizer = TransformerGPT2Tokenizer(cached_dir)
dataset = TopicalDataset(config.dataset_dir, tokenizer)

docs = [doc for doc in dataset]
dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=config.no_below,
                           no_above=config.no_above)

temp = dictionary[0]
id2token = dictionary.id2token
corpus = [dictionary.doc2bow(doc) for doc in docs]



lda_model = LdaModel(
            corpus=corpus,
            id2word=id2token,
            chunksize=config.chunksize,
            alpha=config.alpha,
            eta=config.eta,
            iterations=config.iterations,
            num_topics=config.num_topics,
            passes=config.passes,
            eval_every=config.eval_every
        )


#cm = CoherenceModel(model=lsi_model, corpus=corpus, coherence='u_mass')
cm = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_w2v')
coherence = cm.get_coherence()
print("coherence: ", coherence)
wandb.log({"coherence": coherence})




