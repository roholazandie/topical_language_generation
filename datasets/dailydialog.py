from tokenizers.tokenization_gpt2 import GPT2Tokenizer
import os
import ast

class DailyDialog():

    def __init__(self, dirname):
        model_name_or_path = "gpt2"  # 50257 tokens
        tokenizer_class = GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        self.dirname = dirname


    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for file in files:
                if file.endswith(".json"):
                    with open(root + "/" + file) as fr:
                        for line in fr:
                            conversations = ast.literal_eval(line)
                            yield conversations



if __name__ == "__main__":
    daily_dialog = DailyDialog(dirname="/media/rohola/data/dialog_systems/daily_dialog")
    for i in daily_dialog:
        print(i)