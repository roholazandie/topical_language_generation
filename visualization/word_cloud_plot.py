from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt


def word_cloud(frequencies, file_output):
    '''
    take frequencies dict of word: freq as input
    :param frequencies:
    :return:
    '''
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(frequencies)
    # sizes = np.shape(wordcloud)
    # fig = plt.figure()
    # fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward = False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.tight_layout(pad=0)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.show()
    wordcloud.to_file(file_output)
    return plt


if __name__ == "__main__":
    plt = word_cloud(frequencies={"this": 10, "is": 5, "an": 1, "example": 7})
    plt.show()