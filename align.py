# Modified from https://github.com/davidsandberg/facenet/blob/master/src/align/align_dataset_mtcnn.py

import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import mtcnn
import cv2
from scipy import misc


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('unaligned', type=str,
                        help='Directory with unaligned images')
    parser.add_argument('aligned', type=str,
                        help='Directory with aligned face thumbnails')
    parser.add_argument('--image_size', type=int,
                        help='Image size in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box in pixels.', default=44)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


def main(args):

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, './models/')

    minsize = 20                    # minimum size of face
    threshold = [0.6, 0.7, 0.7]     # three steps's threshold
    factor = 0.709                  # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0

    for cls in os.listdir(args.unaligned):
        output_class_dir = os.path.join(args.aligned, cls)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        folder = os.path.join(args.unaligned, cls)
        for image_name in os.listdir(folder):
            nrof_images_total += 1
            image_path = os.path.join(folder, image_name)
            filename = image_name.split('.')[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    error_message = '{}: {}'.format(image_path, e)
                    print(error_message)
                else:
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
                                index = np.argmax(
                                    bounding_box_size - offset_dist_squared * 2.0)
                                det_arr.append(det[index, :])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if args.detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            misc.imsave(output_filename_n, scaled)
                    else:
                        print('Unable to align "%s"' % image_path)

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))