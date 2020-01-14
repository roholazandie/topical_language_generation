from sklearn.datasets import fetch_20newsgroups

class NewsDataset:

    def __init__(self):
        newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
        newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)
        a=1

    def __iter__(self):
        yield None



if __name__ == "__main__":
    news_dataset = NewsDataset()
