from topical_tokenizers import TransformerGPT2Tokenizer

class NYTimesDataset:

    def __init__(self, filename, tokenizer, do_tokenize=True):
        self.filename = filename
        self.tokenizer = tokenizer
        self.do_tokenize = do_tokenize

    def _process_text(self, text):
        token_ids = self.tokenizer.tokenize(text)
        return token_ids

    def __iter__(self):
        with open(self.filename) as file_reader:
            article = []
            for line in file_reader:
                if line.startswith("URL:"):
                    if article:
                        tokens = self._process_text(" ".join(article))
                        yield tokens
                        article = []
                else:
                    if line.rstrip():
                        article.append(line.rstrip())


if __name__ == "__main__":
    filename = "/media/rohola/data/newyork_articles/nytimes_news_articles.txt"
    cached_dir = "/home/rohola/cached_models"
    tokenizer = TransformerGPT2Tokenizer(cached_dir)
    nytimes_dataset = NYTimesDataset(filename, tokenizer)

    for i, article in enumerate(nytimes_dataset):
        print(i, article)