from collections import Counter, defaultdict


class TopicalDictionary:

    def __init__(self, docs, tokenizer):
        self.dictionary = tokenizer.dictionary
        self.token2id = {v: k for k, v in self.dictionary.items()}
        self.docs = docs
        self.num_docs = len(docs)
        docids = [[self.token2id[w] for w in doc] for doc in docs]
        self.dfs = self._create_dfs(docids)

    # def _create_dfs(self, docids):
    #     self.dfs = defaultdict(int)
    #     for k in self.dictionary.keys():
    #         for doc in docids:
    #             if k in doc:
    #                 self.dfs[k] += 1
    #     return dict(self.dfs)

    def _create_dfs(self, docids):
        self.dfs = defaultdict(int)
        self.dfs = self.dfs.fromkeys(self.dictionary.keys(), 0)  # initialize dictionary with keys and zero
        for doc in docids:
            for i in list(set(doc)):
                self.dfs[i] += 1

        # dfs_token = defaultdict(int)
        # for k in self.dfs:
        #     dfs_token[self.dictionary[k]] = self.dfs[k]

        return dict(self.dfs)

    def apply_to_dictionary(self, good_ids):
        self.dictionary = {k: self.dictionary[k] for k in self.dictionary if k in good_ids}
        self.token2id = {v: k for k, v in self.dictionary.items()}

    def filter_extremes(self, no_below=20, no_above=0.5):
        no_above_abs = int(no_above * self.num_docs)
        good_ids = [i for i in self.dictionary.keys() if (no_below <= self.dfs[i] <= no_above_abs)]
        self.apply_to_dictionary(good_ids)

    def doc2bow(self, document):
        document_ids = [self.token2id[w] for w in document if w in self.token2id.keys()]
        document_bow = list(dict(Counter(document_ids)).items())
        #document_bow = sorted(document_bow, key=lambda k: k[0])
        return document_bow
