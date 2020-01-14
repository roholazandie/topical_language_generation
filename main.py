# Compute bigrams.
from gensim.models import Phrases
from preprocessing import tokenize_docs
from gensim.corpora import Dictionary
from gensim.models import LdaModel

docs = tokenize_docs()

# # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
# bigram = Phrases(docs, min_count=20)
# for idx in range(len(docs)):
#     for token in bigram[docs[idx]]:
#         if '_' in token:
#             # Token is a bigram, add to document.
#             docs[idx].append(token)


# Remove rare and common tokens.
# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)