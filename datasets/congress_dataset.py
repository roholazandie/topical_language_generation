from datasets.dataset import Dataset
import os
import logging
from topical_tokenizers import TransformerGPT2Tokenizer
logging.basicConfig(level=logging.DEBUG)

class CongressDataset(Dataset):

    def __init__(self, dirname, tokenizer, do_tokenize=True):
        super().__init__(dirname, tokenizer, do_tokenize)
        logging.debug("using congress dataset")
        self.dirname = dirname
        self.tokenizer = tokenizer
        self.do_tokenize = do_tokenize

    def _process_text(self, text):
        token_ids = self.tokenizer.tokenize(text)
        return token_ids

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for file in files:
                with open(root + "/" + file) as file_reader:
                    for line in file_reader:
                        if line.strip():
                            if self.do_tokenize:
                                yield self._process_text(line)
                            else:
                                yield line



if __name__ == "__main__":
    dirname = "/media/rohola/data/sample_texts/republicans/"
    cached_dir = "/home/rohola/cached_models"
    tokenizer = TransformerGPT2Tokenizer(cached_dir)

    congress_dataset = CongressDataset(dirname, tokenizer)
    for doc in congress_dataset:
        print(doc)