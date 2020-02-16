import json


class LDAConfig:

    def __init__(self,
                 name="",
                 dataset_dir="",
                 cached_dir="",
                 dict_dir="",
                 topics_file="",
                 topic_top_words_file="",
                 no_below="",
                 no_above="",
                 tokenizer="",
                 alpha="auto",
                 eta="auto",
                 num_topics="",
                 chunksize="",
                 passes="",
                 iterations="",
                 eval_every=""
                 ):
        self.name = name
        self.dataset_dir = dataset_dir
        self.cached_dir = cached_dir
        self.dict_dir = dict_dir
        self.topics_file = topics_file
        self.topic_top_words_file = topic_top_words_file
        self.no_below = no_below
        self.no_above = no_above
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

    def __str__(self):
        return str(self.__dict__)


class LDAWikiConfig:

    def __init__(self,
                 dataset_dir="",
                 cached_dir="",
                 topics_file="",
                 topic_top_words_file="",
                 no_below="",
                 no_above="",
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
        self.topics_file = topics_file
        self.topic_top_words_file = topic_top_words_file
        self.no_below = no_below
        self.no_above = no_above
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
        config = LDAWikiConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))



class LSIWikiConfig:

    def __init__(self,
                 dataset_dir="",
                 cached_dir="",
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
        config = LSIWikiConfig()
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
                 name="",
                 dataset_dir="",
                 cached_dir="",
                 dict_dir="",
                 topics_file="",
                 topic_top_words_file="",
                 no_below="",
                 no_above="",
                 tokenizer="",
                 num_topics="",
                 chunksize="",
                 passes="",
                 iterations="",
                 eval_every=""
                 ):
        self.name = name
        self.dataset_dir = dataset_dir
        self.cached_dir = cached_dir
        self.dict_dir = dict_dir
        self.topics_file = topics_file
        self.topic_top_words_file = topic_top_words_file
        self.no_below = no_below
        self.no_above = no_above
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

    def __str__(self):
        return str(self.__dict__)

class GenerationConfig:
    def __init__(self, model_type="",
                 model_name_or_path="",
                 cached_dir="",
                 padding_text="",
                 max_length="",
                 temperature="",
                 repetition_penalty="",
                 top_k="",
                 top_p="",
                 fusion_method="",
                 logit_threshold="",
                 gamma="",
                 no_cuda="",
                 seed="",
                 stop_token="",
                 num_beams=""):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.cached_dir = cached_dir
        self.padding_text = padding_text
        self.max_length = max_length
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.fusion_method = fusion_method
        self.logit_threshold = logit_threshold
        self.gamma = gamma
        self.no_cuda = no_cuda
        self.seed = seed
        self.stop_token = stop_token
        self.num_beams = num_beams

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


class DatabaseConfig:
    def __init__(self, database_name="",
                 collection_name="",
                 dataset_dir=""):
        self.database_name = database_name
        self.collection_name = collection_name
        self.dataset_dir = dataset_dir

    @classmethod
    def from_dict(cls, json_object):
        config = DatabaseConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))


class PlotConfig:
    def __init__(self,
                 dimension="",
                 threshold="",
                 node_size="",
                 color_scale="",
                 title="",
                 out_file_name=""
                 ):
        self.dimension = dimension
        self.threshold = threshold
        self.node_size = node_size
        self.color_scale = color_scale
        self.title = title
        self.out_file_name = out_file_name

    @classmethod
    def from_dict(cls, json_object):
        config = PlotConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))