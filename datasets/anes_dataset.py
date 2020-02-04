from datasets.dataset import Dataset
import os
import pandas as pd
import logging
from topical_tokenizers import TransformerGPT2Tokenizer
logging.basicConfig(level=logging.DEBUG)

class AnesDataset(Dataset):

    def __init__(self, dirname, tokenizer, do_tokenize=True):
        super().__init__(dirname, tokenizer, do_tokenize)
        logging.debug("using anes dataset")
        self.dirname = dirname
        self.tokenizer = tokenizer
        self.do_tokenize = do_tokenize

    def _process_text(self, text):
        token_ids = self.tokenizer.tokenize(text)
        return token_ids

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for file in files:
                anes = pd.read_csv(root + "/" + file, sep='\t')
                for index, row in anes.iterrows():
                    col_texts = []
                    for col_name in anes.columns.values:
                        if type(row[col_name]) is str:
                            col_texts.append(row[col_name])

                    text = " ".join(col_texts).replace("<br/>", "")
                    yield self._process_text(text)



if __name__ == "__main__":
    dirname = "/media/rohola/data/ANES-elfardy-mdiab-ccb-dataset/SEM-Split/"
    cached_dir = "/home/rohola/cached_models"
    tokenizer = TransformerGPT2Tokenizer(cached_dir)

    anes_dataset = AnesDataset(dirname=dirname, tokenizer=tokenizer)
    for doc in anes_dataset:
        print(doc)