import spacy


class Metrics:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def distinct_1(self, text):
        tokens = [token.text for token in self.nlp(text)]
        one_grams = set(tokens)
        dist_1 = len(one_grams)/len(tokens)
        return dist_1

    def distinct_2(self, text):
        tokens = [token.text for token in self.nlp(text)]
        bigrams = set(zip(*[tokens[i:] for i in range(2)]))
        dist2 = len(bigrams)/(len(tokens)-1)
        return dist2

    def distinct_3(self, text):
        tokens = [token.text for token in self.nlp(text)]
        trigrams = set(zip(*[tokens[i:] for i in range(3)]))
        dist3 = len(trigrams)/(len(tokens)-2)
        return dist3


class Perplexity:

    def __init__(self):
        pass


if __name__ == "__main__":
    metrics = Metrics()
    #text = "Yes, you will get distinct words (though punctuation will affect all of this to a degree). To generate sentences, I assume that you want something like a Markov chain? I actually wrote up an article on word generation using markov chains a few years ago. The basic ideas are the same: ohthehugemanatee.net/2009/10/…. You'll need to find a way to label starting words in the data structure which this does not do, as well as ending or terminal words. "
    text = "this is a very simple example example example"
    dist1 = metrics.distinct_1(text)
    dist2 = metrics.distinct_2(text)
    dist3 = metrics.distinct_3(text)
    print(dist1)
    print(dist2)
    print(dist3)