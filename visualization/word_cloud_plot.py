from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt


def word_cloud(frequencies):
    '''
    take frequencies dict of word: freq as input
    :param frequencies:
    :return:
    '''
    wordcloud = WordCloud().generate_from_frequencies(frequencies)
    # sizes = np.shape(wordcloud)
    # fig = plt.figure()
    # fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    return plt


if __name__ == "__main__":
    plt = word_cloud(frequencies={"this": 10, "is": 5, "an": 1, "example": 7})
    plt.show()