"""Definitions of some network architectures/handy functions relating to using
neural nets with gym"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


# batch norm constants
NO_BATCH_NORM = 0
BATCH_NORM_TRAIN = 1
BATCH_NORM_TEST = 2


# helpers
def get_input_for(env, batch_size, random_projection=False):
    """gets an input placeholder for observations from a given environment"""
    shape = [batch_size] + list(env.observation_space.shape)

    input_pl = tf.placeholder(tf.float32, shape=shape, name='inputs')

    if random_projection:
        # make projection, keep it in tf.variables but out of trainable ones
        # so it gets saved but not trained
        with tf.variable_scope('input_projection'):
            proj = tf.get_variable(
                'projection', shape=[shape[1], random_projection],
                initializer=tf.random_normal_initializer())
            input_pl = tf.matmul(input_pl, proj)
    return input_pl


def sample_action(logits):
    """Adds ops to sample an action with probabilities given by a network
    output.

    Args:
        logits: the linear outputs of the network.

    Returns:
        tensor: an int tensor containing samples actions.
    """
    return tf.multinomial(logits, 1)


def get_net(model, inputs, env):
    """Gets a net from a specified model.

    Doesn't handle continuous action spaces at this stage.

    Args:
        model (str): what kind of net, options thus far (append `-bn` to any to
            use batch normalisation):
            - `mlp`: a fully connected net with relus.

        inputs (tensor): the variable which will hold inputs.
        env (Environment): the gym environment (could only use the output
            space?)
        batch_size (int): how many to do at once.
    """
    num_actions = env.action_space.n
    if model == 'mlp':
        return sample_action(
            fully_connected_net(inputs, 128, 3, num_actions,
                                batch_norm=model.endswith('bn')))
    else:
        raise ValueError('Unknown model: {}'.format(model))


def fully_connected_net(inputs, width, depth, num_outputs, nonlin=tf.nn.relu,
                        batch_norm=NO_BATCH_NORM):
    """A standard fully connected mlp, of specified depth/width.

    Args:
        inputs (tensor): the inputs to the net
        width (int): the number of hidden units at each layer
        depth (int): the total number of layers
        outputs (int): he number of outputs at the final layer
        nonlin (Optional(callable)): nonlinearity applied to hidden units.
            Defaults to ReLU.
        batch_norm (Optional(int)): whether to use batch normalisation.
            0 (default) means no batch norm, 1 means batch norm with batch
            averages (for training) and 2 means batch norm with finalised
            moving averages (for test time).

    Returns:
        tensor:
    """
    if batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = {
            'is_training': True if batch_norm == BATCH_NORM_TRAIN else False}
    else:
        normalizer_fn, normalizer_params = None, None
    with tf.variable_scope('fcn'):
        with slim.arg_scope([slim.fully_connected],
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            biases_initializer=tf.constant_initializer(0.0),
                            activation_fn=nonlin):
            outputs = slim.repeat(inputs, depth-1, slim.fully_connected, width,
                                  scope='layer')
            outputs = slim.fully_connected(outputs, num_outputs,
                                           activation_fn=None)
    return outputs
