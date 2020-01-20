import spacy
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from collections import defaultdict
import pickle
from os import path


class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self):
        raise NotImplemented()

    def encode(self, text):
        raise NotImplemented()

    def save_dict(self):
        raise NotImplemented()

class SpacyTokenizer(Tokenizer):
    def __init__(self, dict_dir, preprocess=False):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        self.dict_dir = dict_dir
        self.preprocess = preprocess
        self._dictionary = defaultdict()
        self.i = 0

    def _dictionary_exist(self):
        return path.isfile(path.join(self.dict_dir, "dict.p"))

    @property
    def dictionary(self):
        self._dictionary = pickle.load(open(path.join(self.dict_dir, "dict.p"), 'rb'))
        return self._dictionary

    def encode(self, text):
        if self.preprocess:
            docs_tokens = [token.text.lower() for token in self.nlp(text) if not token.is_stop]
        else:
            docs_tokens = [token.text for token in self.nlp(text)]

        if not self._dictionary_exist():
            for token in docs_tokens:
                if token not in self._dictionary:
                    self._dictionary[self.i] = token
                    self.i += 1

        return docs_tokens

    def save_dict(self):
        if self._dictionary:
            pickle.dump(self._dictionary, open(path.join(self.dict_dir, "dict.p"), 'wb'))


class TransformerGPT2Tokenizer(Tokenizer):
    def __init__(self, cached_dir):
        super().__init__()
        model_name_or_path = "gpt2"  # 50257 tokens
        tokenizer_class = GPT2TokenizerFast
        #tokenizer_class = GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cached_dir)

    @property
    def dictionary(self):
        #return self.tokenizer.__dict__
        return self.tokenizer.decoder

    def tokenize(self, text):
        # return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.tokenize(text)

    def decode(self, id):
        return self.tokenizer.decode(id)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def save_dict(self):
        pass


if __name__ == "__main__":
    tokenizer = TransformerGPT2Tokenizer(cached_dir="/home/rohola/codes/topical_language_generation/caches/")
    tokens = tokenizer.tokenize("this is a test")
    ids = tokenizer.encode("this is a test")
    print(ids)
    token = tokenizer.decode(ids)
    print(token)
