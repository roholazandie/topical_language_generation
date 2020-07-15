import time
from configs import LDAConfig, GenerationConfig, LSIConfig
from run_generation import generate_unconditional_text
from topical_generation import generate_lda_text, generate_lsi_text, pplm_text, ctrl_text


results = open("timing_results.txt", 'w')

generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
generation_config = GenerationConfig.from_json_file(generation_config_file)
t1 = time.time()
text = generate_unconditional_text(prompt_text="the issue is",
                                   generation_config=generation_config)
t2 = time.time()
print(text)

results.write("gpt: " + str(t2 - t1) + "\n")



lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"

config = LDAConfig.from_json_file(lda_config_file)
generation_config = GenerationConfig.from_json_file(generation_config_file)

t1 = time.time()
text, _, _ = generate_lda_text(prompt_text="The issue is ",
                               selected_topic_index=-1,
                               lda_config=config,
                               generation_config=generation_config,
                               plot=False
                               )
t2 = time.time()
print(text)
results.write("lda: "+ str(t2 - t1) + "\n")

###############LSI
lsi_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lsi_config.json"
generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
lsi_config = LSIConfig.from_json_file(lsi_config_file)
generation_config = GenerationConfig.from_json_file(generation_config_file)

t1 = time.time()
text, _, _ = generate_lsi_text(
    prompt_text="The issue is",
    selected_topic_index=0,
    lsi_config=lsi_config,
    generation_config=generation_config,
    plot=False)
t2 = time.time()
print(text)
print("LSI: ", t2 -t1)
results.write("lsi: " + str(t2 - t1) + "\n")
#############CTRL
# generation_config_file = "/home/rohola/codes/topical_language_generation/configs/ctrl_generation_config.json"
# generation_config = GenerationConfig.from_json_file(generation_config_file)
# t1 = time.time()
# text = ctrl_text(prompt_text="the issue is that",
#           topic="Politics",
#           generation_config=generation_config)
# t2 = time.time()
# print(text)
# #print("CTRL time:", t2-t1)
# results.write("ctrl: " + str(t2 - t1) + "\n")

################PPLM

topics = ["legal", "military", "politics", "religion", "science", "space", "technology"]

generation_config_file = "/home/rohola/codes/topical_language_generation/configs/pplm_generation_config.json"
generation_config = GenerationConfig.from_json_file(generation_config_file)

t1 = time.time()
text = pplm_text(prompt_text="The issue is",
                 topic=topics[2],
                 generation_config=generation_config)
t2 = time.time()
print(text)
#print(t2 -t1)
results.write("pplm: "+ str(t2 - t1) + "\n")

results.close()