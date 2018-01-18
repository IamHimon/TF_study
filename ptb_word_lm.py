from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import tensorflow.contrib.seq2seq as seq2seq
import TF_test.reader as reader

from tensorflow.python.client import device_lib

from util import *
import sys
# get current working directory -- Better to set the PYTHONPATH env variable
current_working_directory = "D:\\Users\humeng\PycharmProjects\TF_study\TF_test\\"
sys.path.append(current_working_directory)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model', 'small', 'A type of model. Possible options are: small, medium, large.')
flags.DEFINE_string('data_path', 'E:\\NLP\simple-examples\data', 'Where the training/test data is stored.')
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
            # 构造一个[vocab_size,size]的随机值向量，用来表示训练数据中的所有word，
            # 向量的横向维度是size (=hidden_size)，也就是隐层的LSTM cell数量
            embedding = tf.get_variable('embedding', [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
            # inputs形状：(len(input_),size)，将输入数据跟embedding对应起来，找到word对应的那个向量

        # 训练时，需要加上dropout，
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        # 类别数就是词典的数量，每个word看做一个‘类’。
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=data_type())

        # 过一层全连接
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss,
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
        tvars = tf.trainable_variables()    # return all variables created with "Trainable = True"
        # 计算所有变量的梯度，并且根据max_grad_norm来切割过大的梯度来防止梯度爆炸
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        # 用之前的learning rate来起始梯度下降优化器
        optimazer = tf.train.GradientDescentOptimizer(self._lr)

        # 执行梯度，将梯度加入到变量上，还有做global_step的自增
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
        """
        Build the inference graph using canonical LSTM cells.
        :param inputs:
        :param config:
        :param is_training:
        :return:
        """
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

        self._initial_state = cell.zero_state(config.batch_size, data_type())   # Return zero-filled state tensor(s)
        state = self._initial_state

        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # Run this multi-layer cell on inputs, starting from state.
                # 这边是一个word一个word输入
                (cell_output, state) = cell(inputs[:, time_step, :], state)  # 更新state
                outputs.append(cell_output)
        # 把outputs中所有cell_output链接起来，然后reshape成横向维度为hidden_size
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return outputs, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """
        将Graph中所有的op重新命名，然后对应name存储以来，就可以输出来了
        :param name:
        :return:
        """
        self._name = name
        ops = {with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)  # 将op以及name对应的存储起来
        self._initial_state_name = with_prefix(self._name, "initial")   # 重新命名
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

            # # 新版本改变
            # if self._cell and rnn_params:
            #     params_saveable = tf.contrib.cudnn_rnn.CudnnLSTMSaveable(
            #         rnn_params,
            #         self._cell.num_layers,
            #         self._cell.num_units,
            #         self._cell.input_size,
            #         self._cell.input_mode,
            #         self._cell.direction,
            #         scope="Model/RNN")

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
    init_scale = 0.1    # 相关参数的初始化为随机均匀分布，范围是[-init_scale,+init_scale]
    learning_rate = 1.0     # 学习率
    max_grad_norm = 5   # 用于控制梯度膨胀，削减梯度。如果梯度向量的L2模超过max_grad_norm，则等比例缩小
    num_layers = 2  # LSTM层数
    num_steps = 20   # 分隔句子的粒度大小，每次会把num_steps个单词划分为一句话
    hidden_size = 200   # 隐层单元数目，每个词会表示成[hidden_size]大小（维度）的向量
    max_epoch = 4   # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
    max_max_epoch = 13   # 完整的文本要循环的次数
    keep_prob = 1.0      # dropout率，1.0为不丢弃.训练时dropout设定为小于1的值。
    lr_decay = 0.5       # 学习速率衰减指数
    batch_size = 20     # 和num_steps共同作用，把原始训练数据划分为batch_size组，每组划分为n个长度为num_steps的句子
    vocab_size = 10000  # 单词数量(这份训练数据中单词刚好10000种)
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


def get_config():
    if FLAGS.model == "small":
        config = SmallConfig
    elif FLAGS.model == "medium":
        config = MediumConfig
    elif FLAGS.model == "large":
        config = LargeConfig
    elif FLAGS.model == "test":
        config = TestConfig
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    # tf:1.3.0
    # if FLAGS.rnn_model:
    #     config.rnn_mode = FLAGS.rnn_mode
    # if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0":
    #     config.rnn_mode = BASIC
    return config


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    # tf:1.3.0
    # gpus = [
    #     x.name for x in device_lib.list_local_devices() if x.device_typy == "GPU"
    # ]
    #
    # if FLAGS.num_gpus > len(gpus):
    #     raise ValueError(
    #         "Your machine has only %d gpus "
    #         "which is less than the requested --num_gpus=%d."
    #         % (len(gpus), FLAGS.num_gpus))

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

        models = {"Train": m, "Valid": mvalid, "Test": mtest}
        for name, model in models.items():
            model.export_ops(name=name)

        metagraph = tf.train.export_meta_graph()    # Returns `MetaGraphDef` proto

        if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
            raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                             "below 1.1.0")

        # 多gpu并行运算
        soft_placement = False
        if FLAGS.num_gpus > 1:
            soft_placement = True
            auto_parallel(metagraph, m)

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)   # Recreates a Graph saved in a `MetaGraphDef` proto
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                # 递减learning rate
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f " % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)  # 训练模型
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s" % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == '__main__':
    tf.app.run()



'''
@ 关于  tf.get_variable_scope().reuse_variables() 的解释：
这个重要的variable_scope函数的目的其实是允许我们在保留模型权重的情况下运行多个模型。首先，从RNN的根源上说，
因为输入输出有着时间关系，我们的模型在训练时每此迭代都要运用到之前迭代的结果，所以如果我们直接使用
(cell_output, state) = cell(inputs[:, time_step, :], state)我们可能会得到一堆新的RNN模型，
而不是我们所期待的前一时刻的RNN模型。再看main函数，当我们训练时，我们需要的是新的模型，所以我们在定义了
一个scope名为model的模型时说明了我们不需要使用以存在的参数，因为我们本来的目的就是去训练的。而在我们做
validation和test的时候呢？训练新的模型将会非常不妥，所以我们需要运用之前训练好的模型的参数来测试他们的效果，
故定义reuse=True。
'''





