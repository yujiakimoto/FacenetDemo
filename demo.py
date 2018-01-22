import cv2
import pickle
import numpy as np
import tensorflow as tf
import mtcnn
import argparse
import sys
from facenet import Facenet


def detect_faces(img, args):

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, './models/')

    minsize = 20                    # minimum size of face
    threshold = [0.6, 0.7, 0.7]     # three steps's threshold
    factor = 0.709                  # scale factor

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bounding_boxes, _ = mtcnn.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if args.detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        faces = []
        boxes = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = cv2.resize(cropped, (args.image_size, args.image_size))
            boxes.append(bb)
            faces.append(scaled)
        return faces, boxes


def predict(sess, net, classifier, face, names):
    img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    embedding = sess.run(net.end_points['embeddings'],
                         feed_dict={net.end_points['inputs']: np.array([img])})
    pred = classifier.predict_proba(embedding)
    idx = np.argmax(pred)
    prob = pred[0, idx]
    name = names[idx]
    return prob, name


def draw_corners(frame, box):
    length = 20
    width = 2
    colour = (0, 0, 255)
    p1 = np.array([box[0], box[1]])
    p2 = np.array([box[2], box[1]])
    p3 = np.array([box[0], box[3]])
    p4 = np.array([box[2], box[3]])
    left = np.array([-1 * length, 0])
    right = np.array([length, 0])
    up = np.array([0, -1 * length])
    down = np.array([0, length])
    cv2.line(frame, tuple(p1), tuple(p1 + right), colour, width)
    cv2.line(frame, tuple(p1), tuple(p1 + down), colour, width)
    cv2.line(frame, tuple(p2), tuple(p2 + left), colour, width)
    cv2.line(frame, tuple(p2), tuple(p2 + down), colour, width)
    cv2.line(frame, tuple(p3), tuple(p3 + right), colour, width)
    cv2.line(frame, tuple(p3), tuple(p3 + up), colour, width)
    cv2.line(frame, tuple(p4), tuple(p4 + left), colour, width)
    cv2.line(frame, tuple(p4), tuple(p4 + up), colour, width)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int,
                        help='Display window height in pixels', default=720)
    parser.add_argument('--width', type=int,
                        help='Display window width in pixels', default=960)
    parser.add_argument('--image_size', type=int,
                        help='Image size in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box in pixels.', default=44)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':

    args = parse_arguments(sys.argv[1:])
    print('Loading SVM...')
    with open('./models/classifier.pkl', 'rb') as file:
        classifier = pickle.load(file)
    names = np.load('./data/names.npy')
    print('Loading Facenet...')
    net = Facenet()
    cv2.namedWindow('camera')
    cv2.namedWindow('info')
    vc = cv2.VideoCapture(0)
    vc.set(3, args.width)
    vc.set(4, args.height)

    if vc.isOpened():
        ret_val, frame = vc.read()
    else:
        ret_val = False

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        while ret_val:
            cv2.imshow('camera', frame)
            ret_val, frame = vc.read()

            key = cv2.waitKey(20)
            # exit on ESC, show info on ENTER
            if key == 13:
                faces, boxes = detect_faces(frame, args)
                for box in boxes:
                    draw_corners(frame, box)
                for face in faces:
                    prob, name = predict(sess, net, classifier, face, names)
                    if prob < 0.75:
                        cv2.putText(frame, 'Unkown',
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    else:
                        cv2.putText(frame, 'P: {} C: {}'.format(name, round(100 * prob, 1)),
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow('info', frame)
            elif key == 27:
                break

        cv2.destroyWindow('camera')
        vc.release()
