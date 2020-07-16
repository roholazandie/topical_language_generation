import matplotlib.pyplot as plt

times = []
gpt_times = []
lda_times = []
lsi_times = []
ctrl_times = []
pplm_times = []

with open("results/timings.txt") as fr:
    for line in fr:
        items = [float(item) for item in line.split(',')]
        times.append(items[0])
        gpt_times.append(items[1])
        lda_times.append(items[2])
        lsi_times.append(items[3])
        ctrl_times.append(items[4])
        pplm_times.append(items[5])

with plt.style.context('seaborn'):
    #fig = plot_figure(style_label=style_label)
    plt.plot(times, gpt_times, label='GPT2 (Uncoditional)')
    plt.plot(times, lda_times, label='LDA+TLG')
    plt.plot(times, lsi_times, label='LSI+TLG')
    plt.plot(times, ctrl_times, label='CTRL')
    plt.plot(times, pplm_times, label='PPLM')
    plt.xlabel("Number of Generated Tokens")
    plt.ylabel("Time (seconds)")
    plt.legend(loc="upper left")

plt.savefig("times.png")
plt.show()
