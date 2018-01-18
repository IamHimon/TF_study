from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Py3 = sys.version_info[0] == 3


def _read_words(filename):
    """
    读文件中所有的word
    :param filename:
    :return:所有word的list
    """
    with tf.gfile.GFile(filename, 'r') as f:
        if Py3:
            return f.read().replace('\n', '<eos>').split()
        else:
            return f.read().decoder('utf-8').replace('\n', '<eos>').split()


def _build_vocab(filename):
    data = _read_words(filename)
    conter = collections.Counter(data)  # 统计文件中word词频
    # print(conter)
    conter_pairs = sorted(conter.items(), key=lambda x: (-x[1], x[0]))  # 按照word词频的逆序排序
    # print(zip(*conter_pairs))

    words, _ = list(zip(*conter_pairs))  # words:所有的word, _:words对应的所有词频
    # print(words)
    # print(_)
    word_to_id = dict(zip(words, range(len(words))))  # 构造word的词典，word:indexID,(从0开始)
    # print(word_to_id)
    return word_to_id


def _file_to_word_dis(filename, word_to_id: dict):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """
    Load PTB raw data from data directory "data_path".
    :param data_path:
    :return:tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
    """
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_dis(train_path, word_to_id)
    valid_data = _file_to_word_dis(valid_path, word_to_id)
    test_data = _file_to_word_dis(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """
    返回数据和对应的标签
    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    :param raw_data: one of the raw data from ptb_raw_data.[w1.w2,w3,,,,]
    :param batch_size: 要将数据分为多少批
    :param num_steps: the number of unrolls,就是num_steps个单词组成一句话
    :param name:the name of this operation (optional).
    :return:

    Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size  # 每批有多少个
        # 上面用的是整除‘//’，所以会多对几个word，然后raw_data[0:batch_size * batch_len]是去掉这几个
        data = tf.reshape(raw_data[0:batch_size * batch_len], [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps ")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        # 切割,strided_slice(data, start, end), [start, end)，前闭后开
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        # 下面构造target，每个当前word的target（‘类别’或者'ground truth'）就是他的下一个word，语言模型！
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y


if __name__ == '__main__':
    _build_vocab('data/data_part')



