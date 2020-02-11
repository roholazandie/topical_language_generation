import streamlit as st
import json
from configs import LDAConfig, LSIConfig, PlotConfig, GenerationConfig
from lda_model import LDAModel
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
from visualization.word_cloud_plot import word_cloud
from gui.utils import convert_topical_model_names, convert_decoding_algorithm_names, convert_to_dataset_name, create_streamlit_config_file
from topical_generation import generate_lda_text, generate_lsi_text
import pickle
import os

st.title("Topical Language Generation")

st.sidebar.title("Topical Settings:")
dataset_name = st.sidebar.selectbox("Select Dataset:", ["Alexa Topical",
                                                       "Newsgroup",
                                                       "New York Times",
                                                       "ANES",
                                                       "Congress Debates"])

dataset = convert_to_dataset_name(dataset_name)

topic_model_name = st.sidebar.selectbox("Topic Model:", ["Latent Dirichlet Allocation", "Latent Semantic Indexing"])
topic_model = convert_topical_model_names(topic_model_name)

num_topics = int(st.sidebar.text_input("Number of Topics:", 10))
alpha = float(st.sidebar.text_input("Alpha: ", 0.001))

#config = find_default_values(dataset, topic_model)
###################################################
st.sidebar.title("Generation Settings:")
decoding_algorithm_name = st.sidebar.selectbox("Decoding Algorithm:", ["No algorithm", "prenorm"])
decoding_algorithm = convert_decoding_algorithm_names(decoding_algorithm_name)


gamma_value = float(st.sidebar.text_input("Gamma: ", 5))
logit_threshold_value = float(st.sidebar.text_input("Logit Threshold: ", -100))

temperature = float(st.sidebar.text_input("Temperature: ", 1))
repition_penalty = float(st.sidebar.text_input("Repetition Penalty: ", 1))
length = int(st.sidebar.text_input("Generated Text Length: ", 50))

###########################################
config_file = "configs/streamlit_config.json"
print(dataset)
create_streamlit_config_file(config_file, topic_model, dataset)

topic_words = []

streamlit_caches_dir = "/home/rohola/codes/topical_language_generation/caches/streamlit_caches/"
topic_words_file = os.path.join(streamlit_caches_dir, "topic_words.p")
fig_file = os.path.join(streamlit_caches_dir, "fig.p")
if st.button("Plot Topics"):
    with st.spinner('Please be patient ...'):
        lda = LDAModel(config_file=config_file)
        # first apply the GUI parameters to the defualt config
        lda.config.num_topics = num_topics
        lda.config.alpha = alpha
        json.dump(lda.config.__dict__, open(config_file, 'w'))
        # now start
        lda._start()
        all_topic_tokens = lda.get_all_topic_tokens(num_words=15)

        # clean up words
        topic_words = [[(t[0].strip('Ġ'), t[1]) for t in tw] for tw in all_topic_tokens]

        pickle.dump(topic_words, open(topic_words_file, 'wb'))

        plot_config = PlotConfig.from_json_file("configs/plot_config.json")

        fig = visualize_semantic_netwrok(plot_config,
                                         topic_words,
                                         auto_open=False)
        st.plotly_chart(fig)
        pickle.dump(fig, open(fig_file, 'wb'))
else:
    with st.spinner('Waiting ...'):
        try:
            fig = pickle.load(open(fig_file, 'rb'))
            st.plotly_chart(fig)
        except:
            pass

#if topic_words:
if not topic_words:
    try:
        topic_words = pickle.load(open(topic_words_file, 'rb'))
    except:
        topic_words = []

topic_words_show = [[t[0] for t in tw] for tw in topic_words]
selected_topic = st.selectbox("Select Topic:", topic_words_show)
if selected_topic:
    i = topic_words_show.index(selected_topic)
    topic_words = [dict(tw) for tw in topic_words]
    word_cloud(frequencies=topic_words[i])
    st.pyplot()

prompt = st.text_input("Start writing here: ", "The issue is")

if st.button("Generate"):
    #todo get the color of the chosen topic and then use that color for words that are in the topic with prob>some_threshold
    # print(prompt)
    # html = "<span style='color: blue'>var</span> foo = <span style='color: green'>'bar'</span>"
    # html = html.replace("\n", " ")
    # st.write(html, unsafe_allow_html=True)

    #todo generation is probably have different best configs for lda and lsi
    generation_config_file = "configs/generation_config.json"
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    generation_config.length = length
    generation_config.temperature = temperature
    generation_config.repetition_penalty = repition_penalty
    generation_config.method = decoding_algorithm
    generation_config.gamma = gamma_value
    generation_config.logit_threshold = logit_threshold_value
    json.dump(generation_config.__dict__, open(generation_config_file, 'w'))

    text = ""
    if topic_model == "lda":
        text = generate_lda_text(prompt_text=prompt,
                          lda_config_file=config_file,
                          generation_config_file=generation_config_file)


    elif topic_model == "lsi":
        text = generate_lsi_text(prompt_text=prompt,
                                 lsi_config_file=config_file,
                                 generation_config_file=generation_config_file)

    st.write(text)