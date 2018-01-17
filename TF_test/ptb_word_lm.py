from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import TF_test.reader as reader

from tensorflow.python.client import device_lib

from TF_test.util import *

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model', 'small', 'A type of model. Possible options are: small, medium, large.')
flags.DEFINE_string('data_path', None, 'Where the training/test data is stored.')
flags.DEFINE_string('save_path', None, 'Model output directory.')
flags.DEFINE_bool('use_fp16', False, 'Train using 16-bit floats instead of 32bit floats')
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = 'basic'
CUDNN = 'cudnn'
BLOCK = 'block'


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float16


class PTBInput(object):
    """the input data"""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)


class PTBModel(object):
    """the PTB model"""

    def __init__(self, is_training, config, input_: PTBInput):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size  # 字典大小

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config, is_training)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=data_type())

        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        # learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)

        # 用之前的learning rate来起始梯度下降优化器
        optimazer = tf.train.GradientDescentOptimizer(self._lr)

        self._train_op = optimazer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")

        self._lr_update = tf.assign(self._lr, self._new_lr)  # 将_new_lf赋值给_lr

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size], tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size], tf.float32)
        self._initial_state = (rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        # hidden_size可以理解为每一层LSTM cell的个数
        if config.rnn_mode == BASIC:
            return rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.

        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                # 为此lstm cell的输入输出加入dropout机制
                cell = rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state

        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # Run this multi-layer cell on inputs, starting from state.
                (cell_output, state) = cell(inputs[:, time_step, :], state)  # 更新state
                outputs.append(cell_output)
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return outputs, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        self._name = name
        ops = {with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = with_prefix(self._name, "initial")
        self._final_state_name = with_prefix(self._name, "final")

        export_state_tuples(self._initial_state, self._initial_state_name)
        export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)

            self._cost = tf.get_collection_ref(with_prefix(self._name, "cost"))[0]
            num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
            self._initial_state = import_state_tuples(self._initial_state,
                                                      self._initial_state_name, num_replicas)
            self._final_state = import_state_tuples(self._final_state,
                                                    self._final_state_name, num_replicas)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def run_epoch(session, model: PTBModel, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
                   (time.time() - start_time)))
    return np.exp(costs / iters)


