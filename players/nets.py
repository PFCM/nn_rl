"""Definitions of some network architectures"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


# helpers
def get_input_for(env, batch_size, discrete_policy='sparse'):
    """gets an input placeholder for observations from a given environment"""
    pass


def get_net(model, inputs, env):
    """Gets a net from a specified model.

    Args:
        model (str): what kind of net, options tbc.
        inputs (tensor): the variable which will hold inputs.
        env (Environment): the gym environment (could only use the output
            space?)
        batch_size (int): how many to do at once.
    """
    pass


def fully_connected_net(inputs, width, depth, outputs, nonlin=tf.nn.relu):
    """A standard fully connected mlp, of specified depth/width.

    Args:
        inputs (tensor):
        width (int):
        depth (int):
        outputs (int):
        nonlin (Optional(callable)):

    Returns:
        tensor:
    """
    with tf.variable_scope('fcn'):
        with slim.arg_scope(slim.fully_connected,
                            normalizer_fn=None,
                            biases_initializer=tf.constant_initializer(0.0),
                            activation_fn=nonlin):
            outputs = slim.repeat(inputs, depth-1, slim.fully_connected, width,
                                  scope='layer')
            outputs = slim.fully_connected(outputs, activation_fn=None)
    return outputs
