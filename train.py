import numpy as np
import tensorflow as tf
import os
import argparse
import sys
import pickle
from scipy.misc import imread
from facenet import Facenet
from sklearn.svm import SVC


def load_data(path):
    X, y = [], []
    n_people = len(os.listdir(path))
    names = np.empty(n_people, dtype=str)
    idx = 0
    for person in os.listdir(path):
        directory = os.path.join(path, person)
        imgs = os.listdir(directory)
        for imgPath in imgs:
            img = imread(os.path.join(directory, imgPath))
            X.append(img)
            y.append(idx)
        names[idx] = person
        idx += 1
    X = np.array(X)
    y = np.array(y)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_images', type=str,
                        help='Directory with aligned face thumbnails',
                        default='./data/aligned')
    args = parser.parse_args(sys.argv[1:])
    main(args)