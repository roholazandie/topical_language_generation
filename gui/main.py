import streamlit as st
import json
from configs import LDAConfig, LSIConfig, PlotConfig, GenerationConfig
from lda_model import LDAModel
from lsi_model import LSIModel
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
from visualization.word_cloud_plot import word_cloud
from gui.utils import convert_topical_model_names, convert_decoding_algorithm_names, convert_to_dataset_name, get_draft_config
from topical_generation import generate_lda_text, generate_lsi_text
from gui.SessionState import get
from PIL import Image
import pickle
import os
import sys

#os.environ['PATH'] += ':'+ '/home/rohola/codes/topical_language_generation/utils/'


session_state = get(dataset="",
                    topic_model="",
                    num_topics="",
                    config="",
                    generation_config="",
                    alpha="",
                    fusion_algorithm="",
                    gamma_value="",
                    logit_threshold_value="",
                    temperature="",
                    repition_penalty="",
                    length="",
                    seed_number="",
                    topic_words="",
                    fig="",
                    i="",
                    )

st.title("Topical Language Generation (TLG)")

st.sidebar.title("Topical Settings:")
dataset_name = st.sidebar.selectbox("Select Dataset:", ["Alexa Topical",
                                                       "Newsgroup",
                                                       "New York Times",
                                                       "ANES",
                                                       "Congress Debates"])

session_state.dataset = convert_to_dataset_name(dataset_name)

topic_model_name = st.sidebar.selectbox("Topic Model:", ["Latent Dirichlet Allocation", "Latent Semantic Indexing"])
session_state.topic_model = convert_topical_model_names(topic_model_name)

session_state.num_topics = int(st.sidebar.text_input("Number of Topics:", 10))

if session_state.topic_model == "lda":
    session_state.alpha = float(st.sidebar.text_input("Alpha: ", 0.001))

###################################################
st.sidebar.title("Generation Settings:")
decoding_algorithm_name = st.sidebar.selectbox("Decoding Algorithm:", ["No algorithm", "prenorm"])
session_state.fusion_algorithm = convert_decoding_algorithm_names(decoding_algorithm_name)


session_state.gamma_value = float(st.sidebar.text_input("Gamma: ", 4))
session_state.logit_threshold_value = float(st.sidebar.text_input("Logit Threshold: ", -95))

session_state.temperature = float(st.sidebar.text_input("Temperature: ", 1))
session_state.repition_penalty = float(st.sidebar.text_input("Repetition Penalty: ", 1.2))
session_state.length = int(st.sidebar.text_input("Generated Text Length: ", 50))

session_state.seed_number = int(st.sidebar.text_input("Seed Number: ", 41))

topic_words = []

if st.button("Plot Topics"):
    with st.spinner('Please be patient ...'):
        if session_state.topic_model == "lda":
            session_state.config = get_draft_config(session_state.topic_model, session_state.dataset)
            session_state.config.num_topics = session_state.num_topics
            session_state.config.alpha = session_state.alpha
            lda = LDAModel(session_state.config, build=True)
            all_topic_tokens = lda.get_all_topic_tokens(num_words=15)

            a = lda.get_psi_matrix()
            print("first time", a.max())

            # clean up words
            session_state.topic_words = [[(t[0].strip('Ġ'), t[1]) for t in tw] for tw in all_topic_tokens]
            plot_config = PlotConfig.from_json_file("configs/lda_plot_config.json")

            fig = visualize_semantic_netwrok(plot_config,
                                             session_state.topic_words,
                                             auto_open=False)
            st.plotly_chart(fig)
            session_state.fig = fig
        elif session_state.topic_model == "lsi":
            session_state.config = get_draft_config(session_state.topic_model, session_state.dataset)
            session_state.config.num_topics = session_state.num_topics
            lsi = LSIModel(session_state.config, build=True)

            tw = lsi.get_topic_words(num_words=10)
            topic_words = [t[1] for t in tw]
            # clean up words
            session_state.topic_words = [[(t[0].strip('Ġ'), t[1]) for t in tw] for tw in topic_words]

            plot_config = PlotConfig.from_json_file("configs/lsi_plot_config.json")
            fig = visualize_semantic_netwrok(plot_config,
                                             session_state.topic_words,
                                             auto_open=False)
            st.plotly_chart(fig)
            session_state.fig = fig

else:
    with st.spinner('Waiting ...'):
        try:
            st.plotly_chart(session_state.fig)
        except:
            pass


topic_words_show = [[t[0] for t in tw] for tw in session_state.topic_words]
selected_topic = st.selectbox("Select Topic:", topic_words_show)

if selected_topic:
    session_state.i = topic_words_show.index(selected_topic)
    topic_words = [dict(tw) for tw in session_state.topic_words]
    topic_words = [{key: abs(tw[key]) for key in tw} for tw in topic_words] #we need absolute values rather than just scores with negative values
    image_file = 'cloud.png'
    word_cloud(frequencies=topic_words[session_state.i], file_output=image_file)
    cloud_image = Image.open(image_file)
    st.image(cloud_image, caption='Word Cloud', use_column_width=True)

prompt = st.text_input("Start writing here: ", "The issue is")

if st.button("Generate"):

    #todo generation is probably have different best configs for lda and lsi
    generation_config_file = "configs/generation_config.json"
    #streamlit_generation_config_file = "configs/st_generation_config.json"
    session_state.generation_config = GenerationConfig.from_json_file(generation_config_file)
    session_state.generation_config.length = session_state.length
    session_state.generation_config.temperature = session_state.temperature
    session_state.generation_config.repetition_penalty = session_state.repition_penalty
    session_state.generation_config.fusion_method = session_state.fusion_algorithm
    session_state.generation_config.gamma = session_state.gamma_value
    session_state.generation_config.logit_threshold = session_state.logit_threshold_value
    session_state.generation_config.seed = session_state.seed_number
    #json.dump(generation_config.__dict__, open(streamlit_generation_config_file, 'w'))

    text = ""
    if session_state.topic_model == "lda":
        with st.spinner('Thinking in LDA...'):
            text, tokens, token_weights = generate_lda_text(prompt_text=prompt,
                                     selected_topic_index=session_state.i,
                                     lda_config=session_state.config,
                                     generation_config=session_state.generation_config)


    elif session_state.topic_model == "lsi":
        with st.spinner('Thinking in LSI ...'):
            print(session_state.config)
            text, tokens, token_weights = generate_lsi_text(prompt_text=prompt,
                                         selected_topic_index=session_state.i,
                                         lsi_config=session_state.config,
                                         generation_config=session_state.generation_config)



    final_text = ""
    for token, token_weight in zip(tokens, token_weights):
        final_text += "<span style='background-color: rgba(255, 0, 0, "+str(token_weight)+")'>"+token+"</span> "

    st.write(final_text, unsafe_allow_html=True)

    st.write("<hr>", unsafe_allow_html=True)
    st.write(text)