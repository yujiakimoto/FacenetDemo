import pygame
import pygame.camera
import tensorflow as tf
import numpy as np
import facenet
import os
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from tkinter import Tk, Label, Button, PhotoImage
import tkinter.font as font

class IntroducerGUI:
    def __init__(self, master, classifier_args=None, info=None):
        self.master = master
        self.args = classifier_args
        self.classifier = None
        self.info = info
        self.master.title('Welcome to CDS!')
        self.font = font.Font(family='Helvetica', size=11)

        self.label = Label(self.master, width=40, text='')
        self.label.config(font=('Helvetica', 12))
        self.label.grid(row=0, column=1)

        self.name_label = Label(self.master, text='')
        self.name_label.config(font=('Helvetica', 12))
        self.name_label.grid(row=1, column=1)

        self.major_label = Label(self.master, text='')
        self.major_label.config(font=('Helvetica', 12))
        self.major_label.grid(row=2, column=1)

        self.fact_label = Label(self.master, text='')
        self.fact_label.config(font=('Helvetica', 12))
        self.fact_label.grid(row=3, column=1)

        self.model_status = Label(self.master, text='Status: Untrained')
        self.model_status.config(font=('Helvetica', 10))
        self.model_status.grid(row=0, column=0)

        self.train_button = Button(self.master, text='Train FaceNet', width=20, height=5, font=self.font, command=self.initialize_classifier)
        self.train_button.grid(row=1, column=0)

        self.start_button = Button(self.master, text='Start Camera', width=20, height=5, font=self.font, command=self.start_camera)
        self.start_button.grid(row=2, column=0)

        self.close = Button(master, text='Quit', width=20, height=5, command=self.master.quit)
        self.close.grid(row=3, column=0)

    def initialize_classifier(self):
        self.classifier = kevin_facey(args=self.args, GUI=self)
        self.classifier.train()
        self.model_status['text'] = 'Status: Trained'

    def start_camera(self):
        cam = webcam(classifier=self.classifier)

    def update_text(self):
        # print(self.classifier.output)
        self.name_label['text'] = 'Name: ' + str(self.classifier.output)
        self.major_label['text'] = 'Major: ' + self.info[self.classifier.output]['Major']
        self.fact_label['text'] = 'Role: ' + self.info[self.classifier.output]['Role']
        self.master.update()


class kevin_facey:

    def __init__(self, args, GUI):
        self.model = None
        self.class_names = []
        self.GUI = GUI
        self.output = 'Unknown'

        with tf.Graph().as_default() as self.graph:

            self.sess = tf.Session()
            with self.sess.as_default():

                np.random.seed(seed=args['seed'])
                self.dataset = facenet.get_dataset(args['data_dir'])
                self.paths, self.labels = facenet.get_image_paths_and_labels(self.dataset)
                # print(self.paths, self.labels)

                print('Number of classes: %d' % len(self.dataset))
                print('Number of images: %d' % len(self.paths))

                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(args['model'])

                # Get input and output tensors
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(self.paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args['batch_size']))
                self.emb_array = np.zeros((nrof_images, self.embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * args['batch_size']
                    end_index = min((i + 1) * args['batch_size'], nrof_images)
                    self.paths_batch = self.paths[start_index:end_index]
                    images = facenet.load_data(self.paths_batch, False, False, args['image_size'])
                    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                    self.emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

                self.classifier_filename_exp = os.path.expanduser(args['classifier_filename'])

    def train(self):
        # Train classifier
        print('Training classifier')
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(self.emb_array, self.labels)

        # Create a list of class names
        self.class_names = [cls.name.replace('_', ' ') for cls in self.dataset]

        # Saving classifier model
        with open(self.classifier_filename_exp, 'wb') as outfile:
            pickle.dump((self.model, self.class_names), outfile)
        print('Saved classifier model to file "%s"' % self.classifier_filename_exp)

    def predict(self, test_dir):
        # Classify images
        # print('Testing classifier')

        with self.sess.as_default():

            np.random.seed(seed=args['seed'])
            self.dataset = facenet.get_dataset(test_dir)
            self.paths, _ = facenet.get_image_paths_and_labels(self.dataset)

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(self.paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args['batch_size']))
            self.emb_array = np.zeros((nrof_images, self.embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args['batch_size']
                end_index = min((i + 1) * args['batch_size'], nrof_images)
                self.paths_batch = self.paths[start_index:end_index]
                images = facenet.load_data(self.paths_batch, False, False, args['image_size'])
                feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                self.emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

        predictions = self.model.predict_proba(self.emb_array)
        # print(predictions)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        self.output = self.class_names[best_class_indices[0]]
        self.GUI.update_text()

        for i in range(len(best_class_indices)):
            print(self.paths[i], '%s: %.3f' % (self.class_names[best_class_indices[i]], best_class_probabilities[i]))

class webcam:

    def __init__(self, session=None, classifier=None):

        self.size = (640, 480)
        self.events = None
        self.ready = False
        display = pygame.display.set_mode(self.size, 0)
        pygame.camera.init()
        cam = pygame.camera.Camera(pygame.camera.list_cameras()[0], self.size)
        cam.start()
        snapshot = pygame.surface.Surface(self.size, 0, display)

        try:
            while True:
                self.events = pygame.event.get()
                snapshot = cam.get_image(snapshot)
                display.blit(snapshot, (0, 0))
                pygame.display.flip()
                for event in self.events:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        img = cam.get_image()
                        pygame.image.save(img, os.path.join('test', 'images', 'img.jpg'))
                        classifier.predict('test/')
                        # raw = img.get_buffer().raw
                        # arr = np.flip(np.frombuffer(raw, dtype=np.ubyte).reshape(self.size[1], self.size[0], 3), 2)
        except KeyboardInterrupt:
            pass
        cam.stop()

if __name__ == '__main__':

    batch_size = 1000
    classifier_filename = '/home/yuji/models/facenet/trial1.pkl'
    data_dir = '/home/yuji/datasets/cds/training/'
    image_size = 160
    model = '/home/yuji/models/facenet/20170512/20170512.pb'
    seed = 666
    args = dict(((k, eval(k)) for k in ['batch_size','classifier_filename','data_dir','image_size','model','seed']))

    CDS = dict()
    CDS['Ryan Butler'] = {'Major':'CS (\'19)', 'Role':'Technology Lead'}
    CDS['Dae Won Kim'] = {'Major':'ORIE (MEng \'18)', 'Role':'Senior Advisor'}
    CDS['Chase Thomas'] = {'Major':'InfoSci (\'19)', 'Role':'President'}
    CDS['Jared Lim'] = {'Major':'CS (\'20)', 'Role':'Education Lead'}
    CDS['Amit Mizrahi'] = {'Major':'CS (\'19)', 'Role':'Events Lead'}
    CDS['Jo Chuang'] = {'Major':'CS (\'19)', 'Role':'Project Manager: Kaggle'}
    CDS['Kenta Takatsu'] = {'Major':'CS (\'19)', 'Role':'Project Manager: Yelp'}
    CDS['Michael Druyan'] = {'Major':'CS (\'19)', 'Role':'Co-Project Manager: Algo Trading'}
    CDS['Neil Shah'] = {'Major':'ORIE (\'19)', 'Role':'Co-Project Manager: Algo Trading'}

    root = Tk()
    my_gui = IntroducerGUI(root, classifier_args=args, info=CDS)
    root.mainloop()

    # cam = webcam(classifier=classifier)
    # classifier.predict('~/datasets/trials/testing')
    # X_embedded = TSNE().fit_transform(classifier.emb_array)