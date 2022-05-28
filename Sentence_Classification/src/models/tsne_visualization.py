import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot(word_vectors, word_labels, title):
    model_tsne = TSNE(perplexity=40,
                      n_components=2,
                      init='pca',
                      n_iter=2500,
                      random_state=23)
    new_values = model_tsne.fit_transform(word_vectors)
    x, y = [], []

    for val in new_values:
        x.append(val[0])
        y.append(val[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(word_labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    i = 0
    while i < len(x):
        plt.plot(x[i:i + 2], y[i:i + 2])
        i += 2

    plt.title(title)

    return plt


def visualize_word_embedding(wv_model, word_embed_dict, title, vector_size):
    word_labels = []
    word_vectors = []

    for key in word_embed_dict.keys():

        words = word_embed_dict[key]

        if len(words) > 1:
            vec = np.zeros((vector_size, ))
            name = ''
            for w in words:
                try:
                    vec += wv_model.wv[w]  # sum of word vectors
                except:
                    vec += np.zeros((vector_size, ))
                name = name + w.capitalize() + ' '
            vec = vec / len(words)  # mean of word vectors
            word_vectors.append(vec)
            word_labels.append(name + '(' + key[0].capitalize() + ')')
        else:
            try:
                word_vectors.append(wv_model.wv[words[0]])
            except:
                word_vectors.append(np.zeros((vector_size, )))
            word_labels.append(words[0].capitalize() + '(' +
                               key[0].capitalize() + ')')

    plt = tsne_plot(word_vectors, word_labels, title)

    return plt
