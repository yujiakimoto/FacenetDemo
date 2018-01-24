import numpy as np
import tensorflow as tf
import os
import argparse
import sys
import pickle
from scipy.misc import imread
from facenet import Facenet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
    print('Loading FaceNet...')
    facenet = Facenet()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('Generating embeddings...')
        X_emb = sess.run(facenet.end_points['embeddings'],
                         feed_dict={facenet.end_points['inputs']: data['X']})
        # np.save('embeddings.npy', X_emb)

    # cls = SVC(probability=True)
    cls = KNeighborsClassifier(n_neighbors=11)
    print('Training classifier...')
    cls.fit(X_emb, data['y'])

    with open('./models/classifier.pkl', 'wb') as f:
        pickle.dump(cls, f)

    np.save('./sample_data/names.npy', data['names'])

    x_tsne = TSNE().fit_transform(X_emb)
    n_people = np.unique(data['y']).shape[0]
    colours = iter(plt.cm.rainbow(np.linspace(0, 1, n_people)))
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(n_people):
        idx = data['y'] == i
        ax.scatter(x_tsne[idx, 0], x_tsne[idx, 1], c=next(colours), label=data['names'][i])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_images', type=str,
                        help='Directory with aligned face thumbnails',
                        default='./sample_data/aligned')
    args = parser.parse_args(sys.argv[1:])
    main(args)