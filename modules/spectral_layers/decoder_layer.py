from keras import activations, initializers, constraints
from keras import regularizers
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from .graph_ops import graph_conv_op

class Decoder(Layer):
    '''
    Class for for the decoder of the drug structure autoencoder
    '''
    def __init__(self,
                 output_dim,
                 activation='sigmoid',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.output_dim = output_dim

        self.act = activations.get(activation)


    def build(self, input_shape):

        self.input_dim = input_shape[-1]

        self.built = True


    def call(self, inputs):
    
        if len(inputs.get_shape())==3:
            x = tf.transpose(inputs, perm=[0,2,1])
            x = K.batch_dot(inputs, x)
            outputs = self.act(x)
        elif len(inputs.get_shape())==2:
            x = K.transpose(inputs)
            x = K.dot(inputs, x)
            outputs = self.act(x)
    
        else:
            raise ValueError('x must be either 2 or 3 dimension tensor'
                         'Got input shape: ' + str(inputs.get_shape()))            

        return outputs
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.output_dim, self.output_dim)
        return output_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.act),
        }
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
