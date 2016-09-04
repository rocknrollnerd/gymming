import theano
import theano.tensor as T
import lasagne
from lasagne import nonlinearities, objectives, regularization
import numpy as np


class NNGradientPolicy(object):

    def __init__(self, input_size, network_schema, learning_rate=0.01, verbose=False):
        self.input_size = input_size
        self.network_schema = network_schema
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.input = T.matrix('X')
        self.target = T.matrix('y')
        self.reward = T.vector('r')
        self.build_network()

    def build_network(self):
        self.nn = lasagne.layers.InputLayer((None, self.input_size), self.input)
        for layer_params in self.network_schema:
            self.nn = lasagne.layers.DenseLayer(self.nn, layer_params['n_neurons'], nonlinearity=layer_params['nonlinearity'])

    def act(self, observation):
        observation = np.expand_dims(np.float32(observation), 0)
        action_proba = self.output(observation)
        return action_proba[0]

    def update(self, observations, actions, rewards):
        loss = self.train(np.float32(observations), np.float32(actions), np.float32(rewards))
        if self.verbose:
            print 'loss {:.4f}'.format(float(loss))


class NNCategoricalGradientPolicy(NNGradientPolicy):

    def build_network(self):
        super(NNCategoricalGradientPolicy, self).build_network()
        out = lasagne.layers.get_output(self.nn)
        pg_log_likelihood = T.log(T.sum(out * self.target, axis=1)) * self.reward
        loss = -T.sum(pg_log_likelihood)

        params = lasagne.layers.get_all_params(self.nn, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=self.learning_rate)

        self.train = theano.function([self.input, self.target, self.reward], loss, updates=updates)
        self.output = theano.function([self.input], out)


class NNGaussianGradientPolicy(NNGradientPolicy):

    def build_network(self):
        super(NNGaussianGradientPolicy, self).build_network()
        out = lasagne.layers.get_output(self.nn)

        # gaussian log-likelihood is just negative square error
        pg_log_likelihood = -T.sum(((out - self.target) ** 2), axis=1) * self.reward
        loss = -T.mean(pg_log_likelihood)

        params = lasagne.layers.get_all_params(self.nn, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=self.learning_rate)

        self.train = theano.function([self.input, self.target, self.reward], loss, updates=updates)
        self.output = theano.function([self.input], out)
