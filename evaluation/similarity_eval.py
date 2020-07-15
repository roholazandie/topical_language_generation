from configs import LDAConfig, GenerationConfig
from lda_model import LDAModel
from run_generation import generate_unconditional_text
from topical_generation import generate_document_like_text
from evaluation.similarity_measures import TextSimilarity
import numpy as np


def similarity_measure(config, generation_config, num_docs):
    text_similarity = TextSimilarity()

    nnlm_tlg_similarities = []
    nnlm_gpt_similarities = []

    bert_tlg_similarities = []
    bert_gpt_similarities = []

    gpt_text = generate_unconditional_text(prompt_text="This is a",
                                           generation_config=generation_config)

    for doc_id in range(num_docs):
        if doc_id > 100:
            break

        tlg_text, doc = generate_document_like_text(prompt_text="This is a",
                                                    doc_id=doc_id,
                                                    lda_config=config,
                                                    generation_config=generation_config)

        nnlm_tlg_similarities.append(text_similarity.nnlm_sentence_similarity(tlg_text, doc))
        nnlm_gpt_similarities.append(text_similarity.nnlm_sentence_similarity(gpt_text, doc))

        # bert_tlg_similarities.append(text_similarity.bert_sentence_similarity(tlg_text, doc))
        # bert_gpt_similarities.append(text_similarity.bert_sentence_similarity(gpt_text, doc))

    print("nnlm_tlg_similarities", np.mean(nnlm_tlg_similarities), np.std(nnlm_tlg_similarities))
    print("nnlm_gpt_similarities", np.mean(nnlm_gpt_similarities), np.std(nnlm_gpt_similarities))
    # print("bert_tlg_similarities", np.mean(bert_tlg_similarities), np.std(bert_tlg_similarities))
    # print("bert_gpt_similarities", np.mean(bert_gpt_similarities), np.std(bert_gpt_similarities))


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
