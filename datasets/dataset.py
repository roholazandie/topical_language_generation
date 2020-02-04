

class Dataset:
    def __init__(self, dataset_dir, tokenizer, do_tokenize):
        pass

    def _process_text(self, text):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError