#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

import argparse
import logging

import numpy as np
import torch

from configs import LDAConfig, GenerationConfig, LSIConfig
from pplm.run_pplm import set_generic_model_params, generate_text_pplm, full_text_generation
from lda_model import LDAModel
from lsi_model import LSIModel
from pplm.run_pplm import DISCRIMINATOR_MODELS_PARAMS
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

from visualization.plotly_visualize import barchart, multi_barchart, top_words_prob_plot

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

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


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


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text, {}


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text, {}


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    config_file = "configs/generation_topical_config.json"
    config = GenerationConfig.from_json_file(config_file)

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

    topical_model = "lsi"  # "lda"
    if topical_model == "lda":
        lda_config_file = "configs/alexa_lda_config.json"
        lda_model = LDAModel(lda_config_file)
        theta = lda_model.get_theta_matrix()
        psi = lda_model.get_psi_matrix()
        # theta=None

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
        lsi_config_file = "configs/congress_lsi_config.json"
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

    return text


def generate_lda_text(prompt_text, selected_topic_index, lda_config, generation_config, plot=False):
    generation_config.device = torch.device(
        "cuda" if torch.cuda.is_available() and not generation_config.no_cuda else "cpu")
    generation_config.n_gpu = torch.cuda.device_count()

    set_seed(generation_config)

    # Initialize the model and tokenizer
    try:
        generation_config.model_type = generation_config.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[generation_config.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(generation_config.model_name_or_path)
    model = model_class.from_pretrained(generation_config.model_name_or_path)
    model.to(generation_config.device)

    generation_config.max_length = adjust_length_to_model(generation_config.max_length,
                                                          max_sequence_length=model.config.max_position_embeddings)
    logger.info(generation_config)

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = generation_config.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(generation_config.model_type)
        prompt_text = prepare_input(generation_config, model, tokenizer, prompt_text)
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(generation_config.device)

    lda_model = LDAModel(lda_config)
    # theta = lda_model.get_theta_matrix()
    psi = lda_model.get_psi_matrix()
    theta = None

    output_sequences, total_entropies, token_entropies, kl_divergences, token_weights, all_top_words = model.generate(
        input_ids=encoded_prompt,
        generation_config=generation_config,
        selected_topic_index=selected_topic_index,
        psi=psi,
        theta=theta,
        tokenizer=lda_model.tokenizer,
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: text.find(generation_config.stop_token) if generation_config.stop_token else None]

    tokens = [lda_model.tokenizer.tokenizer.convert_ids_to_tokens(i).strip('Ġ') for i in generated_sequence]
    if plot:
        show_prompt = False
        if show_prompt:
            prompt_padding = [0] * len(encoded_prompt[0])
            total_entropies = prompt_padding + total_entropies
            token_entropies = prompt_padding + token_entropies
            kl_divergences = prompt_padding + kl_divergences
        else:
            tokens = tokens[len(encoded_prompt[0]):]


        # barchart(tokens, total_entropies)
        multi_barchart(tokens, total_entropies, token_entropies, names=["Total Entropy",
                                                                        "Token Entropy"])
        barchart(tokens, kl_divergences)

        top_words_prob_plot(all_top_words)


    return text, tokens, token_weights


def generate_lsi_text(prompt_text, selected_topic_index, lsi_config, generation_config, plot=False):
    generation_config.device = torch.device(
        "cuda" if torch.cuda.is_available() and not generation_config.no_cuda else "cpu")
    generation_config.n_gpu = torch.cuda.device_count()

    set_seed(generation_config)

    # Initialize the model and tokenizer
    try:
        generation_config.model_type = generation_config.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[generation_config.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(generation_config.model_name_or_path)
    model = model_class.from_pretrained(generation_config.model_name_or_path)
    model.to(generation_config.device)

    generation_config.max_length = adjust_length_to_model(generation_config.max_length,
                                                          max_sequence_length=model.config.max_position_embeddings)
    logger.info(generation_config)

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = generation_config.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(generation_config.model_type)
        prompt_text = prepare_input(generation_config, model, tokenizer, prompt_text)
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(generation_config.device)

    lsi_model = LSIModel(lsi_config)
    topic_word_matrix = lsi_model.get_topic_words_matrix()

    output_sequences, total_entropies, token_entropies, kl_divergences, token_weights, all_top_words = model.generate(
        input_ids=encoded_prompt,
        generation_config=generation_config,
        topic_word_matrix=topic_word_matrix,
        selected_topic_index=selected_topic_index,
        tokenizer=lsi_model.tokenizer,
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: text.find(generation_config.stop_token) if generation_config.stop_token else None]

    tokens = [lsi_model.tokenizer.tokenizer.convert_ids_to_tokens(i).strip('Ġ') for i in generated_sequence]
    if plot:
        show_prompt = False
        if show_prompt:
            prompt_padding = [0] * len(encoded_prompt[0])
            total_entropies = prompt_padding + total_entropies
            token_entropies = prompt_padding + token_entropies
            kl_divergences = prompt_padding + kl_divergences
        else:
            tokens = tokens[len(encoded_prompt[0]):]


        # barchart(tokens, total_entropies)
        multi_barchart(tokens, total_entropies, token_entropies, names=["Total Entropy",
                                                                        "Token Entropy"])


        barchart(tokens, kl_divergences)

        top_words_prob_plot(all_top_words)

    return text, tokens, token_weights


def generate_document_like_text(prompt_text, doc_id, lda_config, generation_config):
    lda_model = LDAModel(lda_config)
    theta = lda_model.get_theta_matrix()
    psi = lda_model.get_psi_matrix()

    # get the original doc
    docs = lda_model.get_docs()
    doc = " ".join([t.strip('Ġ') for t in docs[doc_id]])
    # generation_config.max_length = len(doc.split()) # set the max_length to selected doc length

    generation_config.device = torch.device(
        "cuda" if torch.cuda.is_available() and not generation_config.no_cuda else "cpu")
    generation_config.n_gpu = torch.cuda.device_count()

    set_seed(generation_config)

    # Initialize the model and tokenizer
    try:
        generation_config.model_type = generation_config.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[generation_config.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(generation_config.model_name_or_path)
    model = model_class.from_pretrained(generation_config.model_name_or_path)
    model.to(generation_config.device)

    generation_config.max_length = adjust_length_to_model(generation_config.max_length,
                                                          max_sequence_length=model.config.max_position_embeddings)
    logger.info(generation_config)

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = generation_config.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(generation_config.model_type)
        prompt_text = prepare_input(generation_config, model, tokenizer, prompt_text)
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(generation_config.device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        generation_config=generation_config,
        psi=psi,
        theta=theta,
        doc_id=doc_id,
        tokenizer=None,  # lda_model.tokenizer,
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: text.find(generation_config.stop_token) if generation_config.stop_token else None]

    return text, doc


def ctrl_text(prompt_text, topic, generation_config):
    assert generation_config.model_type == "ctrl", "wrong model_type"
    assert generation_config.model_name_or_path == "ctrl", "wrong model_name_or_path"

    generation_config.device = torch.device(
        "cuda:1" if torch.cuda.is_available() and not generation_config.no_cuda else "cpu")
    generation_config.n_gpu = torch.cuda.device_count()

    set_seed(generation_config)
    # Initialize the model and tokenizer
    try:
        generation_config.model_type = generation_config.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[generation_config.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(generation_config.model_name_or_path,
                                                cache_dir=generation_config.cached_dir)
    model = model_class.from_pretrained(generation_config.model_name_or_path,
                                        cache_dir=generation_config.cached_dir)
    model.to(generation_config.device)

    requires_preprocessing = generation_config.model_type in PREPROCESSING_FUNCTIONS.keys()
    prompt_text = topic + " " + prompt_text

    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(generation_config.model_type)
        prompt_text = prepare_input(generation_config, model, tokenizer, prompt_text)
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(generation_config.device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        generation_config=generation_config,
        do_sample=True,
        tokenizer=None,
    )

    # Batch size == 1. to add more examples please use num_return_sequences > 1
    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    # text = text[: text.find(generation_config.stop_token) if generation_config.stop_token else None]

    return text


def pplm_text(prompt_text, topic, generation_config):
    # set Random seed
    torch.manual_seed(generation_config.seed)
    np.random.seed(generation_config.seed)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not generation_config.no_cuda else "cpu"

    discrim = None
    discrim_weights = None
    discrim_meta = None
    num_samples = 1
    class_label = -1

    # generation_config.model_type = "gpt2-medium"

    if discrim == "generic":
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim]["pretrained_model"]
        print("discrim = {}, pretrained_model set " "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = GPT2LMHeadModel.from_pretrained(generation_config.model_type, output_hidden_states=True)
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(generation_config.model_type)

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + prompt_text)

    # print("= Prefix of sentence =")
    # print(tokenizer.decode(tokenized_cond_text))
    # print()

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=topic,
        discrim=discrim,
        class_label=class_label,
        length=generation_config.max_length,
        stepsize=0.03,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=5,
        decay=False,
        gamma=1.5,
        gm_scale=0.99,
        kl_scale=0.01,
        repetition_penalty=generation_config.repetition_penalty,
    )

    pert_gen_text = tokenizer.decode(pert_gen_tok_texts[0].tolist()[0][1:])

    return pert_gen_text


if __name__ == "__main__":
    ##############LDA
    # lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    #
    # config = LDAConfig.from_json_file(lda_config_file)
    # generation_config = GenerationConfig.from_json_file(generation_config_file)
    #
    # text, _, _ = generate_lda_text(prompt_text="The issue is ",
    #                                selected_topic_index=-1,
    #                                lda_config=config,
    #                                generation_config=generation_config,
    #                                plot=True
    #                                )
    # print(text)

    ###############LSI
    # lsi_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lsi_config.json"
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    # lsi_config = LSIConfig.from_json_file(lsi_config_file)
    # generation_config = GenerationConfig.from_json_file(generation_config_file)
    #
    # text, _, _ = generate_lsi_text(
    #                          #prompt_text="Most of the conversation was about ",
    #                          prompt_text="The issue is",
    #                          selected_topic_index=0,
    #                          lsi_config=lsi_config,
    #                          generation_config=generation_config, plot=True)
    #
    # print(text)

    #############CTRL
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/ctrl_generation_config.json"
    # generation_config = GenerationConfig.from_json_file(generation_config_file)
    # text = ctrl_text(prompt_text="the issue is that",
    #           topic="Politics",
    #           generation_config=generation_config)
    #
    # print(text)
    ###############document_like
    from evaluation.similarity_measures import bert_sentence_similarity, calculate_similarity
    from run_generation import generate_unconditional_text
    lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"

    config = LDAConfig.from_json_file(lda_config_file)
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    tlg_text, doc = generate_document_like_text(prompt_text="This is a",
                                       doc_id=320,#173,
                                        lda_config=config,
                                        generation_config=generation_config)

    gpt_text = generate_unconditional_text(prompt_text="This is a",
                                       generation_config=generation_config)



    # print("original: ", doc)
    # print("generated: ", text)
    print("doc and tlg bert", bert_sentence_similarity(doc, tlg_text))
    print("doc and tlg nnlm", calculate_similarity(doc, tlg_text))

    print("doc and gpt bert", bert_sentence_similarity(doc, gpt_text))
    print("doc and gpt nnlm", calculate_similarity(doc, gpt_text))

    ################PPLM
    # import time
    #
    # topics = ["legal", "military", "politics", "religion", "science", "space", "technology"]
    #
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/pplm_generation_config.json"
    # generation_config = GenerationConfig.from_json_file(generation_config_file)
    #
    # t1 = time.time()
    # text = pplm_text(prompt_text="The issue is",
    #             topic=topics[2],
    #             generation_config=generation_config)
    # t2 = time.time()
    # print(text)
    # print(t2-t1)
