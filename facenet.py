import tensorflow as tf
import pickle

with open('./models/weights.pkl', 'rb') as f:
    weights = pickle.load(f, encoding='latin1')


def load_weights(scope, verbose=False):
    stem = 'inceptionresnetv1/'
    endings = ['/weights', '/biases',
               '/batchnorm/moving_mean', '/batchnorm/moving_variance',
               '/batchnorm/beta']
    params = []
    for ending in endings:
        key = stem + scope + ending
        try:
            params.append(weights[key])
        except KeyError:
            if verbose:
                print('No parameter:', scope + ending)
            params.append(None)
    return params


def conv(x, n_filters, size, stride=1, padding='SAME', normalizer_fn=tf.nn.batch_normalization, activation_fn=tf.nn.relu, scope=None):
    with tf.variable_scope(scope):
        weights, bias, mov_mean, mov_var, beta = load_weights(tf.get_variable_scope().name)
        add_bias = (bias is not None)
        if type(size) == int:
            assert(weights.shape == (size, size, x.shape.as_list()[-1], n_filters))
        elif type(size) == list:
            assert(weights.shape == (size[0], size[1], x.shape.as_list()[-1], n_filters))
        W = tf.get_variable('W', initializer=weights)
        x = tf.nn.conv2d(input=x, filter=W, strides=[1, stride, stride, 1], padding=padding)
        if add_bias:
            b = tf.get_variable('b', initializer=bias)
            x += b
        if normalizer_fn:
            x = normalizer_fn(x, mov_mean, mov_var, beta, None, 1e-5)
        if activation_fn:
            x = activation_fn(x)
        return x


def max_pool(x, size, stride=1, padding='SAME', scope=None):
    with tf.variable_scope(scope):
        x = tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding)
        return x


def avg_pool(x, size, stride=1, padding='SAME', scope=None):
    with tf.variable_scope(scope):
        x = tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding)
        return x


def fc(x, n_out, normalizer_fn=tf.nn.batch_normalization, activation_fn=tf.nn.relu, scope=None):
    with tf.variable_scope(scope):
        weights, bias, mov_mean, mov_var, beta = load_weights(tf.get_variable_scope().name)
        add_bias = (bias is not None)
        assert(weights.shape == (x.shape.as_list()[-1], n_out))
        W = tf.get_variable('W', initializer=weights)
        x = tf.matmul(x, W)
        if add_bias:
            b = tf.get_variable('b', initializer=bias)
            x += b
        if normalizer_fn:
            x = normalizer_fn(x, mov_mean, mov_var, beta, None, 1e-5)
        if activation_fn:
            x = activation_fn(x)
        return x


# Inception-ResNet-A
def block35(x, scale=1.0, activation_fn=tf.nn.relu, scope=''):
    """Builds the 35x35 resx block."""
    with tf.variable_scope('block35' + scope):
        with tf.variable_scope('branch_0'):
            conv0 = conv(x, 32, 1, scope='conv2d_1x1')
        with tf.variable_scope('branch_1'):
            conv1_0 = conv(x, 32, 1, scope='conv2d_0a_1x1')
            conv1_1 = conv(conv1_0, 32, 3, scope='conv2d_0b_3x3')
        with tf.variable_scope('branch_2'):
            conv2_0 = conv(x, 32, 1, scope='conv2d_0a_1x1')
            conv2_1 = conv(conv2_0, 32, 3, scope='conv2d_0b_3x3')
            conv2_2 = conv(conv2_1, 32, 3, scope='conv2d_0c_3x3')
        mixed = tf.concat([conv0, conv1_1, conv2_2], 3)
        up = conv(mixed, x.shape.as_list()[3], 1, normalizer_fn=None, activation_fn=None, scope='conv2d_1x1')
        x += scale * up
        if activation_fn:
            x = activation_fn(x)
    return x


# Inception-ResNet-B
def block17(x, scale=1.0, activation_fn=tf.nn.relu, scope=''):
    """Builds the 17x17 resx block."""
    with tf.variable_scope('block17' + scope):
        with tf.variable_scope('branch_0'):
            conv0 = conv(x, 128, 1, scope='conv2d_1x1')
        with tf.variable_scope('branch_1'):
            conv1_0 = conv(x, 128, 1, scope='conv2d_0a_1x1')
            conv1_1 = conv(conv1_0, 128, [1, 7], scope='conv2d_0b_1x7')
            conv1_2 = conv(conv1_1, 128, [7, 1], scope='conv2d_0c_7x1')
        mixed = tf.concat([conv0, conv1_2], 3)
        up = conv(mixed, x.shape.as_list()[3], 1, normalizer_fn=None, activation_fn=None, scope='conv2d_1x1')
        x += scale * up
        if activation_fn:
            x = activation_fn(x)
    return x


# Inception-ResNet-C
def block8(x, scale=1.0, activation_fn=tf.nn.relu, scope=''):
    """Builds the 8x8 resx block."""
    with tf.variable_scope('block8' + scope):
        with tf.variable_scope('branch_0'):
            conv0 = conv(x, 192, 1, scope='conv2d_1x1')
        with tf.variable_scope('branch_1'):
            conv1_0 = conv(x, 192, 1, scope='conv2d_0a_1x1')
            conv1_1 = conv(conv1_0, 192, [1, 3], scope='conv2d_0b_1x3')
            conv1_2 = conv(conv1_1, 192, [3, 1], scope='conv2d_0c_3x1')
        mixed = tf.concat([conv0, conv1_2], 3)
        up = conv(mixed, x.shape.as_list()[3], 1, normalizer_fn=None, activation_fn=None, scope='conv2d_1x1')
        x += scale * up
        if activation_fn:
            x = activation_fn(x)
    return x


def reduction_a(x, k, l, m, n):
    with tf.variable_scope('branch_0'):
        conv0 = conv(x, n, 3, stride=2, padding='VALID', scope='conv2d_1a_3x3')
    with tf.variable_scope('branch_1'):
        conv1_0 = conv(x, k, 1, scope='conv2d_0a_1x1')
        conv1_1 = conv(conv1_0, l, 3, scope='conv2d_0b_3x3')
        conv1_2 = conv(conv1_1, m, 3, stride=2, padding='VALID', scope='conv2d_1a_3x3')
    with tf.variable_scope('branch_2'):
        pool = max_pool(x, 3, stride=2, padding='VALID', scope='maxpool_1a_3x3')
    x = tf.concat([conv0, conv1_2, pool], 3)
    return x


def reduction_b(x):
    with tf.variable_scope('branch_0'):
        conv0 = conv(x, 256, 1, scope='conv2d_0a_1x1')
        conv_1 = conv(conv0, 384, 3, stride=2, padding='VALID', scope='conv2d_1a_3x3')
    with tf.variable_scope('branch_1'):
        conv1 = conv(x, 256, 1, scope='conv2d_0a_1x1')
        conv1_1 = conv(conv1, 256, 3, stride=2, padding='VALID', scope='conv2d_1a_3x3')
    with tf.variable_scope('branch_2'):
        conv2 = conv(x, 256, 1, scope='conv2d_0a_1x1')
        conv2_1 = conv(conv2, 256, 3, scope='conv2d_0b_3x3')
        conv2_2 = conv(conv2_1, 256, 3, stride=2, padding='VALID', scope='conv2d_1a_3x3')
    with tf.variable_scope('branch_3'):
        pool = max_pool(x, 3, stride=2, padding='VALID', scope='maxpool_1a_3x3')
    x = tf.concat([conv_1, conv1_1, conv2_2, pool], 3)
    return x


def read_image(filename):
    img_contents = tf.read_file(filename)
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = 255 * tf.image.convert_image_dtype(img, dtype=tf.float32)
    return tf.expand_dims(img, 0)


def serving_input_fn():
    inputs = {'image_url': tf.placeholder(tf.string, shape=())}
    filename = tf.squeeze(inputs['image_url'])
    image = read_image(filename)
    image = tf.placeholder_with_default(image, shape=[None, 160, 160, 3])

    features = {'image': image}
    return tf.estimator.export.ServingInputReceiver(features, inputs)


class Facenet:
    
    def __init__(self):       

        self.end_points = {}
        inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.end_points['inputs'] = inputs
        image = tf.image.resize_images(inputs, [160, 160])
        
        with tf.variable_scope('normalize'):
            mean, var = tf.nn.moments(image, axes=[1, 2, 3])
            mean = tf.reshape(mean, [-1, 1, 1, 1])
            std = tf.sqrt(var)
            adj = tf.rsqrt(tf.constant(160 * 160 * 3, dtype=tf.float32))
            std_adj = tf.reshape(tf.maximum(std, adj), [-1, 1, 1, 1])
            normalized = tf.multiply(image - mean, 1/std_adj)
        
        # 79 x 79 x 32
        net = conv(normalized, 32, 3, stride=2, padding='VALID', scope='conv2d_1a_3x3')
        self.end_points['conv2d_1a_3x3'] = net
        # 77 x 77 x 32
        net = conv(net, 32, 3, padding='VALID', scope='conv2d_2a_3x3')
        self.end_points['conv2d_2a_3x3'] = net
        # 77 x 77 x 64
        net = conv(net, 64, 3, scope='conv2d_2b_3x3')
        self.end_points['conv2d_2b_3x3'] = net
        # 38 x 38 x 64
        net = max_pool(net, 3, stride=2, padding='VALID', scope='maxpool_3a_3x3')
        self.end_points['maxpool_3a_3x3'] = net
        # 38 x 38 x 80
        net = conv(net, 80, 1, padding='VALID', scope='conv2d_3b_1x1')
        self.end_points['conv2d_3b_1x1'] = net
        # 36 x 36 x 192
        net = conv(net, 192, 3, padding='VALID', scope='conv2d_4a_3x3')
        self.end_points['conv2d_4a_3x3'] = net
        # 17 x 17 x 256
        net = conv(net, 256, 3, stride=2, padding='VALID', scope='conv2d_4b_3x3')
        self.end_points['conv2d_4b_3x3'] = net
        
        # 5 x Inception-ResNet-A
        with tf.variable_scope('repeat'):
            for i in range(5):
                idx = '_' + str(i+1)
                net = block35(net, scale=0.17, scope=idx)
                self.end_points['mixed_5a' + idx] = net
        
        # Reduction-A
        with tf.variable_scope('mixed_6a'):
            net = reduction_a(net, 192, 192, 256, 384)
        self.end_points['mixed_6a'] = net
        
        
        # 10 x Inception-ResNet-B
        with tf.variable_scope('repeat_1'):
            for i in range(10):
                idx = '_' + str(i+1)
                net = block17(net, scale=0.10, scope=idx)
                self.end_points['mixed_6b' + idx] = net
        
        # Reduction-B
        with tf.variable_scope('mixed_7a'):
            net = reduction_b(net)
        self.end_points['mixed_7a'] = net
        
        # 5 x Inception-ResNet-C
        with tf.variable_scope('repeat_2'):
            for i in range(5):
                idx = '_' + str(i+1)
                net = block8(net, scale=0.2, scope=idx)
                self.end_points['mixed_8a' + idx] = net
        
        net = block8(net, activation_fn=None)
        self.end_points['mixed_8b'] = net
        
        self.end_points['prepool'] = net
        net = avg_pool(net, net.shape.as_list()[1], padding='VALID', scope='avgpool_1a_8x8')
        net = tf.layers.flatten(net)
        
        self.end_points['prelogitsflatten'] = net
        net = fc(net, 128, activation_fn=None, scope='bottleneck')
        
        embeddings = tf.nn.l2_normalize(net, 1)
        self.end_points['embeddings'] = embeddings
