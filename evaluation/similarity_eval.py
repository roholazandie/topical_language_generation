from configs import LDAConfig, GenerationConfig
from lda_model import LDAModel
from run_generation import generate_unconditional_text, adjust_length_to_model, MODEL_CLASSES, set_seed, \
    PREPROCESSING_FUNCTIONS
from topical_generation import generate_document_like_text
from evaluation.similarity_measures import TextSimilarity
import numpy as np
import torch


def similarity_measure(config, generation_config, num_docs):
    text_similarity = TextSimilarity()

    nnlm_tlg_similarities = []
    nnlm_gpt_similarities = []

    # gpt_text = generate_unconditional_text(prompt_text="This is a",
    #                                        generation_config=generation_config)

    lda_model = LDAModel(config)
    docs = lda_model.get_docs()

    ###############################
    generation_config.n_gpu = torch.cuda.device_count()
    generation_config.device = torch.device(
        "cuda" if torch.cuda.is_available() and not generation_config.no_cuda else "cpu")

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



    ###############################

    for doc_id in range(num_docs):
        if doc_id > 100:
            break

        doc = " ".join([t.strip('Ġ') for t in docs[doc_id]])

        generation_config.max_length = 2*len(doc.split())

        prompt_text = doc
        #########################################
        generation_config.max_length = adjust_length_to_model(generation_config.max_length,
                                                              max_sequence_length=model.config.max_position_embeddings)
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
        )

        # Batch size == 1. to add more examples please use num_return_sequences > 1
        generated_sequence = output_sequences[0].tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        text = text[: text.find(generation_config.stop_token) if generation_config.stop_token else None]



        ##########################################

        gpt_text = text


        # gpt_text = generate_unconditional_text(prompt_text=doc,
        #                                        generation_config=generation_config)

        gpt_text = gpt_text.split()[len(doc.split()):] # remove the propmt
        gpt_text = " ".join(gpt_text)

        # tlg_text, doc = generate_document_like_text(prompt_text="This is a", # we also can change this
        #                                             doc_id=doc_id,
        #                                             lda_config=config,
        #                                             generation_config=generation_config)
        #
        # nnlm_tlg_similarities.append(text_similarity.nnlm_sentence_similarity(tlg_text, doc))
        nnlm_gpt_similarities.append(text_similarity.nnlm_sentence_similarity(gpt_text, doc))


    #print("nnlm_tlg_similarities", np.mean(nnlm_tlg_similarities), np.std(nnlm_tlg_similarities))
    print("nnlm_gpt_similarities", np.mean(nnlm_gpt_similarities), np.std(nnlm_gpt_similarities))


if __name__ == "__main__":
    lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"

    config = LDAConfig.from_json_file(lda_config_file)
    generation_config = GenerationConfig.from_json_file(generation_config_file)

    lda_model = LDAModel(config, False)
    theta = lda_model.get_theta_matrix()
    num_docs = theta.shape[0]
    print(num_docs)
    similarity_measure(config, generation_config, num_docs)
