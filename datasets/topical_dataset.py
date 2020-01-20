import os
import json
from topical_tokenizers import TransformerGPT2Tokenizer


class TopicalDataset:

    def __init__(self, dirname, tokenizer):
        self.tokenizer = tokenizer
        self.dirname = dirname

    def _process_text(self, text):
        token_ids = self.tokenizer.tokenize(text)
        return token_ids

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(root + "/" + file) as fr:
                    print(file)
                    knowledges = json.load(fr)
                    for key in knowledges:
                        knowledge = knowledges[key]
                        for agent in knowledge:
                            if agent == "config":
                                continue
                            elif agent == "article":
                                for k in knowledge[agent]:
                                    if k != "url":
                                        articles = knowledge[agent][k]
                                        tokens = self._process_text(articles)
                                        # yield list(dict(Counter(ids)).items())
                                        yield tokens
                            else:
                                agent_knowledge = knowledge[agent]
                                for agent_k in agent_knowledge:
                                    kk = agent_knowledge[agent_k]
                                    if "shortened_wiki_lead_section" in kk:
                                        shortened_wiki_lead_section = kk["shortened_wiki_lead_section"]
                                        tokens = self._process_text(shortened_wiki_lead_section)
                                        # yield list(dict(Counter(tokens)).items())
                                        yield tokens
                                    elif "summarized_wiki_lead_section" in kk:
                                        summarized_wiki_lead_section = kk["summarized_wiki_lead_section"]
                                        tokens = self._process_text(summarized_wiki_lead_section)
                                        # yield list(dict(Counter(tokens)).items())
                                        yield tokens
                                    elif "fun_facts" in kk:
                                        fun_facts = ' '.join([fun_fact for fun_fact in kk["fun_facts"]])
                                        tokens = self._process_text(fun_facts)
                                        # yield list(dict(Counter(tokens)).items())
                                        yield tokens

        self.tokenizer.save_dict()


if __name__ == "__main__":
    dirname = "/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/reading_sets/"
    cached_dir = "/home/rohola/cached_models"
    tokenizer = TransformerGPT2Tokenizer(cached_dir)
    topical_dataset = TopicalDataset(dirname, tokenizer)
    for i in topical_dataset:
        print(i)
