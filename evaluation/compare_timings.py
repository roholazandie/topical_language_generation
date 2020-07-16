import time
from configs import LDAConfig, GenerationConfig, LSIConfig
from run_generation import generate_unconditional_text
from topical_generation import generate_lda_text, generate_lsi_text, pplm_text, ctrl_text


results = open("timing_results.txt", 'w')

gpt_times = []
lda_times = []
lsi_times = []
ctrl_times = []
pplm_times = []

for length in [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800]:


    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    t1 = time.time()
    generation_config.max_length = length
    text = generate_unconditional_text(prompt_text="the issue is",
                                       generation_config=generation_config)
    t2 = time.time()
    print(text)

    results.write("gpt: " + str(length) + " " +str(t2 - t1) + "\n")
    gpt_times.append((length, str(t2 - t1)))



    lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"

    config = LDAConfig.from_json_file(lda_config_file)
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    generation_config.max_length = length
    t1 = time.time()
    text, _, _ = generate_lda_text(prompt_text="The issue is ",
                                   selected_topic_index=-1,
                                   lda_config=config,
                                   generation_config=generation_config,
                                   plot=False
                                   )
    t2 = time.time()
    print(text)
    results.write("lda: " + str(length) + " " + str(t2 - t1) + "\n")
    lda_times.append((length, t2 - t1))

    ###############LSI
    lsi_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lsi_config.json"
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    lsi_config = LSIConfig.from_json_file(lsi_config_file)
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    generation_config.max_length = length
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
    results.write("lsi: " + str(length) + " " +str(t2 - t1) + "\n")
    lsi_times.append((length, t2 - t1))
    #############CTRL
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/ctrl_generation_config.json"
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    generation_config.max_length = length
    t1 = time.time()
    text = ctrl_text(prompt_text="the issue is that",
              topic="Politics",
              generation_config=generation_config)
    t2 = time.time()
    print(text)
    #print("CTRL time:", t2-t1)
    results.write("ctrl: " + str(length) + " " +str(t2 - t1) + "\n")
    ctrl_times.append((length, t2 - t1))
    ################PPLM

    topics = ["legal", "military", "politics", "religion", "science", "space", "technology"]

    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/pplm_generation_config.json"
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    generation_config.max_length = length
    t1 = time.time()
    text = pplm_text(prompt_text="The issue is",
                     topic=topics[2],
                     generation_config=generation_config)
    t2 = time.time()
    print(text)
    #print(t2 -t1)
    results.write("pplm: " + str(length) + " " +str(t2 - t1) + "\n")
    pplm_times.append((length, t2-t1))

results.close()

with open("tresult.txt", 'w') as fw:
    for g, ld, ls, ctr, ppl in zip(gpt_times, lda_times, lsi_times, ctrl_times, pplm_times):
        fw.write(str(g[0])+','+str(g[1])+','+str(ld[1])+','+str(ls[1])+','+str(ctr[1])+','+str(ppl[1])+'\n' )