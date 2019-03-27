import collections
import glob
import os
import pickle
import sys
import numpy
import tensorflow as tf
import re
import string
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def GetInputFiles():
    input_path = "F:\Acad\Spring19\CSCI544_NLP\code_hw\HW1\op_spam_training_data"
    # return glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
    return glob.glob(os.path.join(input_path, '*/*/*/*.txt'))


VOCABULARY = collections.Counter()


# ** TASK 1.
def Tokenize(comment):
    """Receives a string (comment) and returns array of tokens."""

    # words = comment.split()
    # words = comment.lower()
    # words = words.translate(str.maketrans('', '', string.digits))
    # words = words.translate(str.maketrans('', '', string.punctuation))
    # words = re.findall(r"[\w']+", words)
    # words = [word for word in words if (len(word) > 1)]  #8.75

    # words = comment.lower()
    # # words = words.translate(str.maketrans('', '', string.digits))
    # # words = words.translate(str.maketrans('', '', string.punctuation))
    # # words = re.split('; |, |\' |\*|\n | |\.', words)
    # # words = re.findall(r"[\w']+", words)
    # #
    # regex = re.compile('[^a-zA-Z]')
    # words = regex.sub(' ', words).split()
    # # words = re.sub("[^\w]", " ", words).split()
    # #
    # # words = [word.split("\\'") for word in words if (len(word) > 1)
    # # & (word not in string.digits) & (word not in string.punctuation)]
    # stop_words =[]
    # if True:
    #     stop_words = open("StopWords.txt", "r").read()
    #     stop_words = re.split("\W+", stop_words)
    # words = [word for word in words if (word not in stop_words) & (len(word) > 1)]

    words = comment.split()
    words_ = []
    regex = re.compile('[^a-zA-Z]')
    for word in words:
        tmp = regex.sub(' ', word.lower()).split()
        for word_ in tmp:
            if len(word_) > 1:
                words_.append(word_)

    return words_

def l2_weighted_regularizer_(scale, net_):
  def l2_we(weights):
    return scale*tf.nn.l2_loss(tf.matmul(net_, weights))
  return l2_we

VARS = {}

# ** TASK 2.
def FirstLayer(net, l2_reg_val, is_training):
    """First layer of the neural network.

    Args:
      net: 2D tensor (batch-size, number of vocabulary tokens),
      l2_reg_val: float -- regularization coefficient.
      is_training: boolean tensor.A

    Returns:
      2D tensor (batch-size, 40), where 40 is the hidden dimensionality.
    """

    ## keep net for Bonus part test
    batch_size, number_of_vocabulary_tokens = net.shape
    net_input = tf.placeholder(tf.float32, [None, number_of_vocabulary_tokens], name='net_input')
    net_input = 1 * net  # to make a copy

    ### ME
    global VARS
    VARS['net'] = net
    net = tf.nn.l2_normalize(net, axis=1)  # ME Preprocess the layer input
    VARS['net_norm'] = net
    net = tf.contrib.layers.fully_connected(net, 40, activation_fn=None,
                                            normalizer_fn=None, biases_initializer=None, scope="fc1")
    VARS['net_norm_y'] = net
    Y = tf.trainable_variables()[0]
    tf.losses.add_loss(l2_reg_val * tf.nn.l2_loss(net), loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    VARS['net_loss'] = net
    net = tf.contrib.layers.batch_norm(net,  is_training=is_training) # ME
    VARS['net_batchnorm'] = net
    net = tf.nn.tanh(net)  # ME
    VARS['net_tanh'] = net

    tmp = EmbeddingL2RegularizationUpdate(Y, net_input, .005, l2_reg_val)
    tmp = EmbeddingL1RegularizationUpdate(Y, net_input, .005, l2_reg_val)
    VARS['net_bonus'] = net

    return net



# ** TASK 2 ** BONUS part 1
def EmbeddingL2RegularizationUpdate(embedding_variable, net_input, learn_rate, l2_reg_val):
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.
    # net_input = net_input / tf.norm(net_input)
    net_input = tf.nn.l2_normalize(net_input, axis=0)
    grad = l2_reg_val * tf.matmul(tf.transpose(net_input), tf.matmul(net_input, embedding_variable))
    embedding_variable_ = embedding_variable - learn_rate * grad

    ## local test  #better to disable when learning
    batch_size, number_of_vocabulary_tokens = net_input.shape
    net_example = numpy.random.binomial(1, .1, (3, number_of_vocabulary_tokens))
    sigma_fnc = l2_reg_val * tf.nn.l2_loss(tf.matmul(net_input, embedding_variable))
    # assert tf.gradients(sigma_fnc, embedding_variable) == grad, "wrong grad in L2"
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf_grad = sess.run(tf.gradients(sigma_fnc, embedding_variable)[0], feed_dict={net_input: net_example})
    my_grad = sess.run(grad, feed_dict={net_input: net_example})
    # differ = numpy.linalg.norm(tf_grad - my_grad)
    # differ = differ / numpy.linalg.norm(tf_grad)
    # print('l2 grad differentage {}'.format(differ))
    print('l2 grad max difference {}'.format(numpy.max(tf_grad - my_grad)))

    return embedding_variable.assign(embedding_variable_)


# ** TASK 2 ** BONUS part 2
def EmbeddingL1RegularizationUpdate(embedding_variable, net_input, learn_rate, l1_reg_val):
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies one regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.
    net_input = tf.nn.l2_normalize(net_input, axis=0)
    sign_inside = tf.sign(tf.matmul(net_input, embedding_variable))
    where = tf.equal(sign_inside, 0)
    # should replace 0's with random in [-1, 1] for an better (not necessarily acute)implementation
    grad = l1_reg_val * tf.matmul(tf.transpose(net_input), sign_inside)
    embedding_variable_ = embedding_variable - learn_rate * grad

    ## local test  #better to disable when learning
    batch_size, number_of_vocabulary_tokens = net_input.shape
    net_example = numpy.random.binomial(1, .1, (3, number_of_vocabulary_tokens))
    sigma_fnc = l1_reg_val * tf.norm(tf.matmul(net_input, embedding_variable), ord=1)
    # assert tf.gradients(sigma_fnc, embedding_variable) == grad, "wrong grad in L2"
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf_grad = sess.run(tf.gradients(sigma_fnc, embedding_variable)[0], feed_dict={net_input: net_example})
    my_grad = sess.run(grad, feed_dict={net_input: net_example})
    # differ = numpy.linalg.norm(tf_grad - my_grad)
    # differ = differ / numpy.linalg.norm(tf_grad)
    # print('l1 grad differentage {}'.format(differ))
    print('l2 grad max difference {}'.format(numpy.max(tf_grad - my_grad)))

    return embedding_variable.assign(embedding_variable_)


# ** TASK 3
def SparseDropout(slice_x, keep_prob=0.5):
    """Sets random (1 - keep_prob) non-zero elements of slice_x to zero.

    Args:
      slice_x: 2D numpy array (batch_size, vocab_size)

    Returns:
      2D numpy array (batch_size, vocab_size)
    """
    # import random
    # slice_x = 1
    # flat_slice_x = numpy.matrix.flatten(slice_x)
    # idx = list(numpy.nonzero(flat_slice_x)[0])
    # rd_idx = random.sample(idx, int((1 - keep_prob)*len(idx)))
    # flat_slice_x[rd_idx] = 0
    # slice_x = flat_slice_x.reshape(slice_x.shape)

    # idx = numpy.nonzero(slice_x)
    # non_zeros = slice_x[idx]
    # len_ = len(non_zeros)
    # idx_rd = random.sample(range(len_), int((1 - keep_prob) * len_))
    # non_zeros[idx_rd] = 0
    # slice_x[idx] = non_zeros
    # slice_x_copy = numpy.copy(slice_x)
    # idx = numpy.nonzero(slice_x)
    # slice_x[idx] = slice_x[idx] * numpy.random.choice([0, 1], size=(len(idx[0]),), p=[1-keep_prob, keep_prob])

    # this gives the exact proportion
    idx = numpy.nonzero(slice_x)
    len_idx = len(idx[0])
    arr = numpy.zeros(len_idx)
    arr[:int(len_idx*keep_prob)] = 1
    numpy.random.shuffle(arr)
    slice_x[idx] *= arr
    # check
    # assert len(numpy.nonzero(slice_x[idx])[0]) == len(slice_x[idx]) * keep_prob , 'Dropout; not right proportion'


    return slice_x


# ** TASK 4
# TODO(student): YOU MUST SET THIS TO GET CREDIT.
# You should set it to tf.Variable of shape (vocabulary, 40).
EMBEDDING_VAR = None


# ** TASK 5
# This is called automatically by VisualizeTSNE.
def ComputeTSNE(embedding_matrix):
    """Projects embeddings onto 2D by computing tSNE.

    Args:
      embedding_matrix: numpy array of size (vocabulary, 40)

    Returns:
      numpy array of size (vocabulary, 2)
    """
    # embedding_matrix = TSNE(n_components=2, n_iter=250,perplexity=5, n_iter_without_progress=10,
    #                         min_grad_norm=1e-2 ).fit_transform(embedding_matrix)
    embedding_matrix = TSNE(n_components=2).fit_transform(embedding_matrix)
    # return embedding_matrix[:, 2]
    return embedding_matrix


# ** TASK 5
# This should save a PDF of the embeddings. This is the *only* function marked
# marked with "** TASK" that will NOT be automatically invoked by our grading
# script (it will be "stubbed-out", by monkey-patching). You must run this
# function on your own, save the PDF produced by it, and place it in your
# submission directory with name 'tsne_embeds.pdf'.
def VisualizeTSNE(sess):
    if EMBEDDING_VAR is None:
        print('Cannot visualize embeddings. EMBEDDING_VAR is not set')
        return
    embedding_mat = sess.run(EMBEDDING_VAR)
    tsne_embeddings = ComputeTSNE(embedding_mat)

    class_to_words = {
        'positive': [
            'relaxing', 'upscale', 'luxury', 'luxurious', 'recommend', 'relax',
            'choice', 'best', 'pleasant', 'incredible', 'magnificent',
            'superb', 'perfect', 'fantastic', 'polite', 'gorgeous', 'beautiful',
            'elegant', 'spacious'
        ],
        'location': [
            'avenue', 'block', 'blocks', 'doorman', 'windows', 'concierge', 'living'
        ],
        'furniture': [
            'bedroom', 'floor', 'table', 'coffee', 'window', 'bathroom', 'bath',
            'pillow', 'couch'
        ],
        'negative': [
            'dirty', 'rude', 'uncomfortable', 'unfortunately', 'ridiculous',
            'disappointment', 'terrible', 'worst', 'mediocre'
        ]
    }

    # TODO(student): Visualize scatter plot of tsne_embeddings, showing only words
    # listed in class_to_words. Words under the same class must be visualized with
    # the same color. Plot both the word text and the tSNE coordinates.
    print('visualization should generate now')

    # vocab_words = numpy.array([k for k,v in TERM_INDEX.items()])
    class_to_words_idx = {}
    class_pts = {}
    # class_words_words = {}
    for class_ in class_to_words.keys():
        # class_to_words_idx[class_] = {word:index for word,index in TERM_INDEX.items() if word in class_to_words[class_]}
        class_to_words_idx[class_] = [TERM_INDEX[word] for word in class_to_words[class_]]
        class_pts[class_] = tsne_embeddings[class_to_words_idx[class_], :]
        # class_words_words[class_] = list(class_to_words_idx[class_].keys()) # the words in class_to_words does not work since order changes in the finding step
    class_colors = {'positive': 'blue', 'negative': 'Orange', 'furniture': 'red', 'location': 'green'}

    fig, ax = plt.subplots()
    scale = 1
    for class_ in class_to_words.keys():
        for i, _ in enumerate(class_pts[class_]):
            x_ = class_pts[class_][i, 0]*scale
            y_ = class_pts[class_][i, 1]*scale
            word_ = class_to_words[class_][i]
            plt.scatter(x_, y_, marker='o', color=class_colors[class_])
            plt.text(x_, y_ + .2, word_, fontsize=6)
            plt.text(x_, y_ - .2, "(" + str(round(x_, 2)) + ', ' + str(round(y_, 2)) + ")", fontsize=6)
    plt.title("coordinates are multiplied by {}".format(scale))
    # plt.show()
    plt.savefig("tsne_embeds_saved.pdf")
    plt.close()


CACHE = {}


def ReadAndTokenize(filename):
    """return dict containing of terms to frequency."""
    global CACHE
    global VOCABULARY
    if filename in CACHE:
        return CACHE[filename]
    comment = open(filename).read()
    words = Tokenize(comment)

    terms = collections.Counter()
    for w in words:
        VOCABULARY[w] += 1
        terms[w] += 1

    CACHE[filename] = terms
    return terms


TERM_INDEX = None


def MakeDesignMatrix(x):
    global TERM_INDEX
    if TERM_INDEX is None:
        print('Total words: %i' % len(VOCABULARY.values()))
        min_count, max_count = numpy.percentile(list(VOCABULARY.values()), [50.0, 99.8])
        TERM_INDEX = {}
        for term, count in VOCABULARY.items():
            if count > min_count and count <= max_count:
                idx = len(TERM_INDEX)
                TERM_INDEX[term] = idx
    #
    x_matrix = numpy.zeros(shape=[len(x), len(TERM_INDEX)], dtype='float32')
    for i, item in enumerate(x):
        for term, count in item.items():
            if term not in TERM_INDEX:
                continue
            j = TERM_INDEX[term]
            x_matrix[i, j] = numpy.log(1+count)  # 1.0  # Try count or log(1+count) numpy.log(1+count)
            # F1 mean; 1:.8415, count: .8419, log(1+count):.8430
    return x_matrix


def GetDataset():
    """Returns numpy arrays of training and testing data."""
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    classes1 = set()
    classes2 = set()
    for f in GetInputFiles():
        class1, class2, fold, fname = f.split('\\')[-4:]
        classes1.add(class1)
        classes2.add(class2)
        class1 = class1.split('_')[0]
        class2 = class2.split('_')[0]

        x = ReadAndTokenize(f)
        y = [int(class1 == 'positive'), int(class2 == 'truthful')]
        if fold == 'fold4':
            x_test.append(x)
            y_test.append(y)
        else:
            x_train.append(x)
            y_train.append(y)

    ### Make numpy arrays.
    x_test = MakeDesignMatrix(x_test)
    x_train = MakeDesignMatrix(x_train)
    y_test = numpy.array(y_test, dtype='float32')
    y_train = numpy.array(y_train, dtype='float32')

    dataset = (x_train, y_train, x_test, y_test)
    with open('dataset.pkl', 'wb') as fout:
        pickle.dump(dataset, fout)
    return dataset


def print_f1_measures(probs, y_test):
    y_test[:, 0] == 1  # Positive
    positive = {
        'tp': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
        'fp': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
        'fn': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
    }
    negative = {
        'tp': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
        'fp': numpy.sum((probs[:, 0] <= 0)[numpy.nonzero(y_test[:, 0] == 1)[0]]),
        'fn': numpy.sum((probs[:, 0] > 0)[numpy.nonzero(y_test[:, 0] == 0)[0]]),
    }
    truthful = {
        'tp': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
        'fp': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
        'fn': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
    }
    deceptive = {
        'tp': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
        'fp': numpy.sum((probs[:, 1] <= 0)[numpy.nonzero(y_test[:, 1] == 1)[0]]),
        'fn': numpy.sum((probs[:, 1] > 0)[numpy.nonzero(y_test[:, 1] == 0)[0]]),
    }

    all_f1 = []
    for attribute_name, score in [('truthful', truthful),
                                  ('deceptive', deceptive),
                                  ('positive', positive),
                                  ('negative', negative)]:
        precision = float(score['tp']) / float(score['tp'] + score['fp'])
        recall = float(score['tp']) / float(score['tp'] + score['fn'])
        f1 = 2 * precision * recall / (precision + recall)
        all_f1.append(f1)
        print('{0:9} {1:.2f} {2:.2f} {3:.2f}'.format(attribute_name, precision, recall, f1))
    print('Mean F1: {0:.4f}'.format(float(sum(all_f1)) / len(all_f1)))


def BuildInferenceNetwork(x, l2_reg_val, is_training):
    """From a tensor x, runs the neural network forward to compute outputs.
    This essentially instantiates the network and all its parameters.

    Args:
      x: Tensor of shape (batch_size, vocab size) which contains a sparse matrix
         where each row is a training example and containing counts of words
         in the document that are known by the vocabulary.

    Returns:
      Tensor of shape (batch_size, 2) where the 2-columns represent class
      memberships: one column discriminates between (negative and positive) and
      the other discriminates between (deceptive and truthful).
    """
    global EMBEDDING_VAR
    EMBEDDING_VAR = None  # ** TASK 4: Move and set appropriately.

    ## Build layers starting from input.
    net = x

    l2_reg = tf.contrib.layers.l2_regularizer(l2_reg_val)

    ## First Layer
    # net = FirstLayer(net, l2_reg_val, is_training)
    net = FirstLayer(net, l2_reg_val, is_training)

    EMBEDDING_VAR = tf.trainable_variables()[0]

    ## Second Layer.
    net = tf.contrib.layers.fully_connected(
        net, 10, activation_fn=None, weights_regularizer=l2_reg)
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = tf.nn.relu(net)

    net = tf.contrib.layers.fully_connected(
        net, 2, activation_fn=None, weights_regularizer=l2_reg)

    return net


def main(argv):
    ######### Read dataset
    x_train, y_train, x_test, y_test = GetDataset()

    ######### Neural Network Model
    x = tf.placeholder(tf.float32, [None, x_test.shape[1]], name='x')
    y = tf.placeholder(tf.float32, [None, y_test.shape[1]], name='y')
    is_training = tf.placeholder(tf.bool, [])

    l2_reg_val = 1e-6  # Co-efficient for L2 regularization (lambda)
    net = BuildInferenceNetwork(x, l2_reg_val, is_training)

    ######### Loss Function
    tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=net)

    ######### Training Algorithm
    learning_rate = tf.placeholder_with_default(
        numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Debuging
    global VARS
    import IPython
    # IPython.embed()
    # Debuging

    def evaluate(batch_x=x_test, batch_y=y_test):
        probs = sess.run(net, {x: batch_x, is_training: False})
        print_f1_measures(probs, batch_y)

    def batch_step(batch_x, batch_y, lr):
        sess.run(train_op, {
            x: batch_x,
            y: batch_y,
            is_training: True, learning_rate: lr,
        })
        # a = 1
        # sess.run(VARS, {x:batch_x, y:batch_y, learning_rate:lr, is_training:True})

    def step(lr=0.01, batch_size=100):
        indices = numpy.random.permutation(x_train.shape[0])
        for si in range(0, x_train.shape[0], batch_size):
            se = min(si + batch_size, x_train.shape[0])
            slice_x = x_train[indices[si:se]] + 0  # + 0 to copy slice
            slice_x = SparseDropout(slice_x)
            batch_step(slice_x, y_train[indices[si:se]], lr)

    lr = 0.05
    print('Training model ... ')
    for j in range(300): step(lr)
    # IPython.embed()
    for j in range(300): step(lr / 2)
    for j in range(300): step(lr / 4)
    # for j in range(30): step(lr)
    VisualizeTSNE(sess)
    print('Results from training:')
    evaluate()



if __name__ == '__main__':
    # tf.random.set_random_seed(0)
    tf.set_random_seed(0)
    main([])