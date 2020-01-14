from tokenizers.tokenization_gpt2 import GPT2Tokenizer
from gensim.models import LdaModel
import numpy as np
from datasets.topical_dataset import TopicalDataset
import pickle

cached_dir = "/home/rohola/cached_models"
model_name_or_path = "gpt2" #50257 tokens
tokenizer_class = GPT2Tokenizer
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cached_dir)


dirname = "/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/reading_sets/"
topical_dataset = TopicalDataset(dirname, tokenizer)


id2word = tokenizer.decoder
######################################
# Set training parameters.
num_topics = 20
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

model = LdaModel(
    corpus=topical_dataset,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(topical_dataset)


np.save("topical_dataset_topics.npy", model.get_topics())
# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

pickle.dump(top_topics, open("top_topical_dataset_topics.p", "wb"))