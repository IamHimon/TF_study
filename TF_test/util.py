import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def export_state_tuples(state_tuples, name):
    for state_tuple in state_tuples:
        tf.add_to_collection(name, state_tuple.c)
        tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tupes, name, num_replicas):
    restored = []
    for i in range(len(state_tupes) * num_replicas):
        c = tf.get_collection_ref(name)[2 * i + 0]
        h = tf.get_collection_ref(name)[2 * i + 1]
        restored.append(rnn.LSTMStateTuple(c, h))
    return tuple(restored)


def with_prefix(prefix, name):
    """Adds prefix to name."""
    return "/".join((prefix, name))