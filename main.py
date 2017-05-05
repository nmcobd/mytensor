import numpy  as nnnn

>>>>>>> ddd
import gzip
from sklearn import preprocessing
from deepnn import DeepNN
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sjwl import *

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):
    print 'Extracting', filename
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number in MNIST image file')
        num_images = _read32(bytestream)[0]
        rows = _read32(bytestream)[0]
        cols = _read32(bytestream)[0]
        # print num_images, rows, cols
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)

        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    print 'Extracting', filename
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('lll')
        num_items = _read32(bytestream)[0]
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


def train_svc():
    train_data = extract_images('train-images-idx3-ubyte.gz')
    train_labels = extract_labels('train-labels-idx1-ubyte.gz')

    test_data = extract_images('t10k-images-idx3-ubyte.gz')
    test_labels = extract_labels('t10k-labels-idx1-ubyte.gz')
    print train_data.shape
    print train_labels.shape
    normed_train_data = preprocessing.normalize(train_data)
    normed_test_data = preprocessing.normalize(test_data)


    print 'training...'
    svc = SVC(kernel='linear', C=1)
    svc.fit(normed_train_data[:5000, :], train_labels[:5000])
    print 'predicting...'
    pre = svc.predict(normed_test_data[:1000])

    right = 0
    for i in range(len(pre)):
        if pre[i] == test_labels[i]:
            right += 1
    print float(right)/len(pre)


def dnn():

    train_data = extract_images('train-images-idx3-ubyte.gz')
    train_labels = extract_labels('train-labels-idx1-ubyte.gz', False)
    # train_labels.shape = (-1, 10)
    test_data = extract_images('t10k-images-idx3-ubyte.gz')
    test_labels = extract_labels('t10k-labels-idx1-ubyte.gz', False)
    # test_labels.shape = (-1, 10)
    X_train = normalize(train_data)
    X_test = normalize(test_data)
    # print test_labels[:10]
    # svm = SVC(kernel='linear', C=100)
    # svm.fit(X_train, train_labels)
    # pre = svm.predict(X_test)
    # print accuracy_score(test_labels, pre)
    # dnn = DeepNN([600], [500, 64, 10], learning_rate=0.1, lambda_reg=0.001, nb_epoches=1000)

    # dnn.train_dnn(normed_train_data[:1000], train_labels[:1000])
    # dnn.predict(normed_test_data, test_labels)

    sj = SJWL(X_train, train_labels, X_test, test_labels)
    sj.train()
if __name__ == '__main__':
    dnn()
