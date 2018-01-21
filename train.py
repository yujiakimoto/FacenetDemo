import numpy as np
import tensorflow as tf
import os
import argparse
import sys
import pickle
from scipy.misc import imread
from facenet import Facenet
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_data(path):
    X, y, names = [], [], []
    idx = 0
    for person in os.listdir(path):
        directory = os.path.join(path, person)
        imgs = os.listdir(directory)
        for imgPath in imgs:
            img = imread(os.path.join(directory, imgPath))
            X.append(img)
            y.append(idx)
        names.append(person.replace('_', ' '))
        idx += 1
    X = np.array(X)
    y = np.array(y)
    names = np.array(names)
    return {'X': X, 'y': y, 'names': names}


def main(args):
    data = load_data(args.training_images)
    facenet = Facenet()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('Generating embeddings...')
        X_emb = sess.run(facenet.end_points['embeddings'],
                         feed_dict={facenet.end_points['inputs']: data['X']})

    cls = SVC()
    print('Training classifier...')
    cls.fit(X_emb, data['y'])

    with open('./models/classifier.pkl', 'wb') as f:
        pickle.dump(cls, f)

    np.save('./data/names.npy', data['names'])

    x_tsne = TSNE().fit_transform(X_emb)
    n_people = np.unique(data['y']).shape[0]
    colours = iter(plt.cm.rainbow(np.linspace(0, 1, n_people)))
    for i in range(n_people):
        idx = data['y'] == i
        plt.scatter(x_tsne[idx, 0], x_tsne[idx, 1], c=next(colours), label=data['names'][i])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_images', type=str,
                        help='Directory with aligned face thumbnails',
                        default='./data/aligned')
    args = parser.parse_args(sys.argv[1:])
    main(args)