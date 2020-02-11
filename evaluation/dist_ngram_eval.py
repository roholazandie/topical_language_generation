import logging

import numpy as np
import torch

from configs import TopicalGenerationConfig
from lda_model import LDAModel
from lsi_model import LSIModel
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#

def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text




PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    #"xlnet": prepare_xlnet_input,
    #"transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


config_file = "../configs/generation_topical_config.json"
config = TopicalGenerationConfig.from_json_file(config_file)

config.n_gpu = torch.cuda.device_count()
config.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")

config.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
config.n_gpu = torch.cuda.device_count()

set_seed(config)

# Initialize the model and tokenizer
try:
    config.model_type = config.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[config.model_type]
except KeyError:
    raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)
model = model_class.from_pretrained(config.model_name_or_path)
model.to(config.device)

config.length = adjust_length_to_model(config.length, max_sequence_length=model.config.max_position_embeddings)
logger.info(config)

prompt_text = input("Model prompt >>> ")

# Different models need different input formatting and/or extra arguments
requires_preprocessing = config.model_type in PREPROCESSING_FUNCTIONS.keys()
if requires_preprocessing:
    prepare_input = PREPROCESSING_FUNCTIONS.get(config.model_type)
    prompt_text = prepare_input(config, model, tokenizer, prompt_text)
encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
encoded_prompt = encoded_prompt.to(config.device)

topical_model = "lda" #"lda"
if topical_model == "lda":
    lda_config_file = "../configs/alexa_lda_config.json"
    lda_model = LDAModel(lda_config_file)
    theta = None#lda_model.get_theta_matrix()
    psi = lda_model.get_psi_matrix()
    #theta=None

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        psi=psi,
        theta=theta,
        tokenizer=lda_model.tokenizer,
        max_length=config.length,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )

elif topical_model == "lsi":
    lsi_config_file = "../configs/congress_lsi_config.json"
    lsi_model = LSIModel(lsi_config_file)
    topic_word_matrix = lsi_model.get_topic_words_matrix()
    output_sequences = model.generate(
        input_ids=encoded_prompt,
        topic_word_matrix=topic_word_matrix,
        tokenizer=lsi_model.tokenizer,
        max_length=config.length,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )



# Batch size == 1. to add more examples please use num_return_sequences > 1
generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
text = text[: text.find(config.stop_token) if config.stop_token else None]

print(text)
