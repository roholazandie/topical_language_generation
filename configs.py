import json


class LDAConfig:

    def __init__(self,
                 dataset_dir="",
                 cached_dir="",
                 dict_dir="",
                 topics_file="",
                 topic_top_words_file="",
                 tokenizer="",
                 alpha="auto",
                 eta="auto",
                 num_topics="",
                 chunksize="",
                 passes="",
                 iterations="",
                 eval_every=""
                 ):
        self.dataset_dir = dataset_dir
        self.cached_dir = cached_dir
        self.dict_dir = dict_dir
        self.topics_file = topics_file
        self.topic_top_words_file = topic_top_words_file
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.eta = eta
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.iterations = iterations
        self.eval_every = eval_every

    @classmethod
    def from_dict(cls, json_object):
        config = LDAConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))


class LSIConfig:
    def __init__(self,
                 dataset_dir="",
                 cached_dir="",
                 dict_dir="",
                 topics_file="",
                 topic_top_words_file="",
                 tokenizer="",
                 num_topics="",
                 chunksize="",
                 passes="",
                 iterations="",
                 eval_every=""
                 ):
        self.dataset_dir = dataset_dir
        self.cached_dir = cached_dir
        self.dict_dir = dict_dir
        self.topics_file = topics_file
        self.topic_top_words_file = topic_top_words_file
        self.tokenizer = tokenizer
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.passes = passes
        self.iterations = iterations
        self.eval_every = eval_every

    @classmethod
    def from_dict(cls, json_object):
        config = LSIConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))


class GenerationConfig:
    def __init__(self, model_type="",
                 model_name_or_path="",
                 cached_dir="",
                 padding_text="",
                 length="",
                 temperature="",
                 repetition_penalty="",
                 top_k="",
                 top_p="",
                 no_cuda="",
                 seed="",
                 stop_token=""):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.cached_dir = cached_dir
        self.padding_text = padding_text
        self.length = length
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.no_cuda = no_cuda
        self.seed = seed
        self.stop_token = stop_token

    @classmethod
    def from_dict(cls, json_object):
        config = GenerationConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))


class TopicalGenerationConfig:
    def __init__(self, model_type="",
                 model_name_or_path="",
                 cached_dir="",
                 padding_text="",
                 length="",
                 temperature="",
                 repetition_penalty="",
                 top_k="",
                 top_p="",
                 no_cuda="",
                 seed="",
                 stop_token=""):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.cached_dir = cached_dir
        self.padding_text = padding_text
        self.length = length
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.no_cuda = no_cuda
        self.seed = seed
        self.stop_token = stop_token
        self.n_gpu = 1
        self.device = 'cpu'

    @classmethod
    def from_dict(cls, json_object):
        config = TopicalGenerationConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))
