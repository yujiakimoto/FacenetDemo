# Facenet Demo
A demo to classify people appearing in webcam using a pre-trained FaceNet network. Implementation is heavily based on
code by [David Sandberg](https://github.com/davidsandberg/facenet). The original paper by Schroff et al. can be found 
[here](https://arxiv.org/pdf/1503.03832.pdf).

## Dependencies
Python 3, TensorFlow, NumPy, SciPy, MatPlotLib   
PyGame, Tkinter (webcam demo only)

## Collect Training Data
Training data can be in any image format (e.g. JPEG, PNG), with images organized as follows:
```
raw_data/
    person_1/
        person1_001.jpg
        person1_002.jpg
        person1_003.jpg
    person_2/
        person2_001.png
        person2_002.png
    etc...
```
Images do not need to be cropped around the face area (see below), can contain multiple people (although it 
should be avoided for simplicity), and can be of any size (as long as the face occupies more than 20x20 pixels).

## Align Training Data
Face detection and alignment is conducted using Zhang's [MTCNN](https://arxiv.org/pdf/1604.02878.pdf). To conduct
alignment on your training set, run:
```
python align.py ./data/raw_data/ ./data/aligned --image_size=160, --margin=44, --detect_multiple_faces=False
```
The first two arguments are the directory containing all training images, and a directory in which to save aligned
images (sub-folders will be created automatically if they do not exist already). The last three arguments are optional
and defaults are listed above.

## Train Classifier (of Embeddings)
To classify faces in a custom dataset, we generate the 128-dimensional embeddings produced by facenet, then train
a separate classifier (in this case a SVM) that maps from the 128-dimensional space to people. To train the SVM, run
the following command:
```
python train.py --training_images=./data/aligned
```
A scatter plot showing a 2D representation (using t-SNE) of the generated embeddings will be shown, and the fitted 
sklearn classifier will be saved as a .pkl file. 

## Run Demo
```
python demo.py
```