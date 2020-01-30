from gensim.corpora import WikiCorpus
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel, TfidfModel
from topical_tokenizers import TransformerGPT2Tokenizer


wiki_raw_file = "/media/rohola/data/raw_wikipedia/enwiki-latest-pages-articles.xml.bz2"
outp = ""
cached_dir = "/home/rohola/codes/topical_language_generation/caches/wiki_cache/"

tokenizer = TransformerGPT2Tokenizer(cached_dir)

wiki = WikiCorpus(wiki_raw_file, tokenizer_func=tokenizer.tokenize)

wiki.dictionary.filter_extremes(no_below=20, no_above=0.1)#this probably should be changed to no_above=0.2 or something above 0.1


MmCorpus.serialize(outp + '_bow.mm', wiki, progress_cnt=10000, metadata=True)  # another ~9h
wiki.dictionary.save_as_text(outp + '_wordids.txt.bz2')
# load back the id->word mapping directly from file
# this seems to save more memory, compared to keeping the wiki.dictionary object from above
dictionary = Dictionary.load_from_text(outp + '_wordids.txt.bz2')

del wiki

# initialize corpus reader and word->id mapping
mm = MmCorpus(outp + '_bow.mm')

# build tfidf, ~50min
tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
tfidf.save(outp + '.tfidf_model')

# save tfidf vectors in matrix market format
# ~4h; result file is 15GB! bzip2'ed down to 4.5GB
MmCorpus.serialize(outp + '_tfidf.mm', tfidf[mm], progress_cnt=10000)