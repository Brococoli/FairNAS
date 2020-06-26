"""
set custom layers which can change inplanes and outplanes
"""
import logging
from tensorflow.keras import layers, Model
import tensorflow as tf


class SuperBatchNormalization(layers.Layer):
    def __init__(self, 
                 max_filters_in,
                 momentum=0.9, 
                 epsilon=0.00001, 
                 center=True, 
                 scale=True,
                 axis=-1,
                 beta_initializer='zeros', 
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros', 
                 moving_variance_initializer='ones',
                 inference_update_stat = False,
                 name=None,
                 **kwargs):
        super(SuperBatchNormalization, self).__init__(name=name, **kwargs)
        self.momentum = momentum
        self.momentum_rest = 1.0 - momentum
        if axis < 0 : axis += 4
        self.axes = [0,1,2,3][:axis]+[0,1,2,3][axis+1:]
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.inference_update_stat = inference_update_stat
        

        self.max_filters_in = max_filters_in

        self.gamma = self.add_weight(name=self.name+'/gamma',
                                      shape=(self.max_filters_in,),
                                      initializer=self.gamma_initializer,
                                      trainable=scale)

        self.beta = self.add_weight(name=self.name+'/beta',
                                      shape=(self.max_filters_in,),
                                      initializer=self.beta_initializer,
                                      trainable=center)

        self.moving_mean = self.add_weight(name=self.name+'/moving_mean',
                                        shape=(self.max_filters_in,),
                                        initializer=self.moving_mean_initializer,
                                        trainable=False)

        self.moving_var = self.add_weight(name=self.name+'/moving_var',
                                        shape=(self.max_filters_in,),
                                        initializer=self.moving_variance_initializer,
                                        trainable=False)
        self.concat = layers.Concatenate(axis=-1, name='cat')
    

    def call(self, x, training=True, **kwargs):
        filters_in = x.shape[-1]
        assert filters_in <= self.max_filters_in
        logging.debug('use bn')

        mean, var = tf.nn.moments(x, axes=self.axes, keepdims=False, name='moments')

        gamma = tf.slice(self.gamma, [0], [filters_in])
        beta = tf.slice(self.beta, [0], [filters_in])

        if not self.inference_update_stat: 
            logging.debug('bn')
            out = tf.nn.batch_normalization(x, mean, var, 
                    offset=beta, scale=gamma, 
                    variance_epsilon=self.epsilon, name='bn')

        else:
            moving_mean = tf.slice(self.moving_mean, [0], [filters_in])
            moving_var = tf.slice(self.moving_var, [0], [filters_in])
                       
            moving_mean = self.momentum * mean + self.momentum_rest * moving_mean
            moving_var = self.momentum * var + self.momentum_rest * moving_var
            self.moving_mean.assign( self.concat([moving_mean, self.moving_mean[x.shape[-1]:]]) )
            self.moving_var.assign( self.concat([moving_var, self.moving_var[x.shape[-1]:]]) )

            out = tf.nn.batch_normalization(x, 
                        moving_mean, moving_var, 
                        offset=beta, scale=gamma, 
                        variance_epsilon=self.epsilon, name='bn')
                
        return out

class SuperConv2d(layers.Layer):
    def __init__(self, 
                 max_filters_in, 
                 max_filters_out,
                 max_kernel_size, 
                 strides=(1, 1), 
                 padding='SAME', 
                 data_format=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None,
                 name=None, **kwargs):
        super(SuperConv2d, self).__init__(name=name, **kwargs)

        self.max_kernel_size = max_kernel_size
        self.max_filters_in = max_filters_in
        self.max_filters_out = max_filters_out
        
        self.strides = [strides]*2 if type(strides) == int else strides
        self.padding=padding
        self.data_format=data_format

        self.use_bias=use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer

        
        self.w = self.add_weight(name=self.name+'/kernel',
                                      shape=(max_kernel_size, max_kernel_size, max_filters_in, max_filters_out),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        if self.use_bias:
            self.b = self.add_weight(name=self.name+'/bias',
                                      shape=(max_filters_out,),
                                      initializer=self.bias_initializer,
                                      trainable=True)
            
    def call(self, x, training=None, filters_out=None, kernel_size=None, **kwargs):
        
        kernel_size = kernel_size if kernel_size is not None else self.max_kernel_size
        filters_out = filters_out if filters_out is not None else self.max_filters_out
        
        filters_in = x.shape[-1]
        assert filters_in <= self.max_filters_in
        assert kernel_size <= self.max_kernel_size, (kernel_size, self.max_kernel_size)
        assert filters_out <= self.max_filters_out
               
        weights = tf.slice(self.w, [0]*4, [kernel_size, kernel_size, filters_in, filters_out])

        conv = tf.nn.conv2d(x, filters=weights, strides=self.strides, 
                            padding=self.padding, data_format=self.data_format,
                            name='conv')

        out = conv
        
        if self.use_bias:
            bias = tf.slice(self.b, [0], [filters_out])
            out = out + bias
        
        if self.kernel_regularizer:
            self.add_loss(self.kernel_regularizer(weights), inputs=True)
        if self.bias_regularizer:
            self.add_loss(self.bias_regularizer(bias), inputs=True)

        return out


class SuperDepthwiseConv2D(layers.Layer):
    def __init__(self, 
                 max_filters_in,
                 max_kernel_size, 
                 strides=[1,1], 
                 padding='SAME', 
                 max_depth_multiplier=1,
                 data_format=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform', 
                 bias_initializer='zeros',
                 depthwise_regularizer=None, 
                 bias_regularizer=None, 
                 name=None, **kwargs):
        super(SuperDepthwiseConv2D, self).__init__(name=name, **kwargs)
    
        strides = [strides]*2 if type(strides) == int else strides
        self.strides = [1] + strides + [1]

        self.max_kernel_size = max_kernel_size
        self.max_depth_multiplier = max_depth_multiplier
        self.max_filters_in = max_filters_in


        
        self.padding=padding
        self.data_format=data_format

        self.use_bias=use_bias
        self.depthwise_initializer=depthwise_initializer
        self.bias_initializer=bias_initializer
        self.depthwise_regularizer=depthwise_regularizer
        self.bias_regularizer=bias_regularizer

        self.w = self.add_weight(name=self.name+'/kernel',
                                      shape=(max_kernel_size, max_kernel_size, max_filters_in, max_depth_multiplier),
                                      initializer=self.depthwise_initializer,
                                      trainable=True)
        if self.use_bias:
            self.b = self.add_weight(name=self.name+'/bias',
                                      shape=(max_filters_in*max_depth_multiplier),
                                      initializer=self.bias_initializer,
                                      trainable=True)
    

    def call(self, x,  training=True, kernel_size=None, depth_multiplier=None, **kwargs):

        
        kernel_size = kernel_size if kernel_size is not None else self.max_kernel_size
        depth_multiplier = depth_multiplier if depth_multiplier is not None else self.max_depth_multiplier

        filters_in = x.shape[-1]
        assert filters_in <= self.max_filters_in
        assert kernel_size <= self.max_kernel_size
        assert depth_multiplier <= self.max_depth_multiplier


            
        weights = tf.slice(self.w, [0]*4, [kernel_size, kernel_size, filters_in, depth_multiplier])


        conv = tf.nn.depthwise_conv2d(
                x, weights, strides=self.strides, padding=self.padding,
                data_format=self.data_format, name='dconv'
                )


        out = conv
        
        if self.use_bias:

            bias = tf.slice(self.b, [0], [filters_in*depth_multiplier])
            out = out + bias
      
        
        if self.depthwise_regularizer:
            self.add_loss(self.depthwise_regularizer(weights), inputs=True)
        if self.bias_regularizer:
            self.add_loss(self.bias_regularizer(bias), inputs=True)
        
        return out

class Activation(layers.Layer):
    def __init__(self, activation, name=None, **kwargs):
        super(Activation, self).__init__(name=name, **kwargs)
        if activation in ['linear', 'relu', 'elu', 'selu', 'softmax', 'sigmoid', 'hard_sigmoid']:
            self.activate = layers.Activation(name, name=name)
        elif activation == 'relu6':
            self.activate = lambda x : tf.nn.relu6(x, name=name)
        elif activation == 'prelu':
            self.activate = layers.PReLU(shared_axes=[1,2], name=name)
        elif activation == 'swish':
            self.activate = tf.nn.swish
        else:
            raise ValueError('Activate name error , donot exist %s' % name)
    def call(self, x, **kwargs):
        x = self.activate(x)
        return x


class SuperMBConvBlock(Model):
    def __init__(self, 
                max_filters_in,
                max_filters_out, 
                max_expand_ratio, 
                max_kernel_size,
                se_ratio, 
                weight_decay, 
                strides , 
                use_shortcut=True,
                drop_connect_rate=None, 
                data_format=None,
                activation='relu6', 
                name=None, **kwargs):
        super(SuperMBConvBlock, self).__init__(name=name, **kwargs)
        
        self.use_shortcut = use_shortcut
        self.max_filters_in = max_filters_in
        self.max_filters_out = max_filters_out
        self.max_expand_ratio = max_expand_ratio
        self.max_kernel_size =  max_kernel_size

        
        strides = [strides]*2 if type(strides) == int else strides
        max_expand_filters = max_filters_in * max_expand_ratio

        if max_expand_ratio != 1:
            self.expand_conv = SuperConv2d(max_filters_in=max_filters_in, 
                 max_filters_out=max_expand_filters,
                 max_kernel_size=1, 
                 strides=(1, 1), 
                 padding='SAME', 
                 data_format=data_format,
                 use_bias=False,
                 kernel_initializer='he_normal', 
                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                 name='expand_conv')
            self.expand_bn = SuperBatchNormalization(max_expand_filters, name='bn')
            self.expand_act = Activation(activation, name = activation)



        #print('filters_in', expand_filters)
        #Depthwise Convlution
        self.extract_dconv = SuperDepthwiseConv2D(max_filters_in=max_expand_filters,
                                     max_kernel_size=max_kernel_size, 
                                     strides=strides, 
                                     padding='SAME', 
                                     max_depth_multiplier=1,
                                     data_format=data_format,
                                     use_bias=False,
                                     depthwise_initializer='he_normal', 
                                     depthwise_regularizer=tf.keras.regularizers.l2(weight_decay), 
                                     name='extract_dconv')
        self.extract_bn = SuperBatchNormalization(max_expand_filters, name='extract_bn')
        self.extract_act = Activation(activation, name=activation)

        
        self.project_conv = SuperConv2d(max_filters_in=max_expand_filters, 
                 max_filters_out=max_filters_out,
                 max_kernel_size=1, 
                 strides=(1, 1), 
                 padding='SAME', 
                 data_format=data_format,
                 use_bias=False,
                 kernel_initializer='he_uniform', 
                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
                 name='project_conv')
        self.project_bn = SuperBatchNormalization(max_filters_out, name='project_bn')
        

        if use_shortcut:
            if drop_connect_rate and (0.0 < drop_connect_rate < 1.0) :
                self.dp = layers.Dropout(drop_connect_rate, noise_shape=(None, 1, 1, 1),
                                   name='dp')
            else:
                self.dp = lambda x, training : x

        
    def call(self, x, training, filters_out=None, expand_ratio=None, kernel_size=None):
            kernel_size = kernel_size if kernel_size is not None else self.max_kernel_size
            filters_out = filters_out if filters_out is not None else self.max_filters_out
            expand_ratio = expand_ratio if expand_ratio is not None else self.max_expand_ratio
            
            filters_in = x.shape[-1]
            assert kernel_size <= self.max_kernel_size
            assert filters_in <= self.max_filters_in
            assert filters_out <= self.max_filters_out
            assert expand_ratio <= self.max_expand_ratio
            
            origin = x
            if self.max_expand_ratio != 1:
                x = self.expand_conv(x, training, kernel_size=1, filters_out=filters_in*expand_ratio)
                x = self.expand_bn(x, training=training)
                x = self.expand_act(x)



            x = self.extract_dconv(x, training, kernel_size=kernel_size, depth_multiplier=1)
            x = self.extract_bn(x, training=training)
            x = self.extract_act(x)

            x = self.project_conv(x, training, filters_out = filters_out, kernel_size=1)
            x = self.project_bn(x, training=training)
            
            if self.use_shortcut and origin.shape[1:] == x.shape[1:]:
                x = self.dp(x, training)
                x = x + origin
            
            return x


class SuperKESMBConvBlock(Model):

    """
    "kernel size and expand ratio select ConvBlock
    """
    def __init__(self, 
                expand_ratio, 
                kernel_size,
                **kwargs):
        super(SuperKESMBConvBlock, self).__init__(name=kwargs['name'])
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size

        #self.choices = [ [[]] * len(expand_ratio) ] * len(kernel_size)
        self.choices = [] 
        for k_size in kernel_size:
            for ex_ratio in expand_ratio:
                self.choices.append( SuperMBConvBlock(max_expand_ratio=ex_ratio, 
                                max_kernel_size=k_size, **kwargs) )
        
    def call(self, x, training,  expand_ratio=None, kernel_size=None, **kwargs):
        
        assert kernel_size in self.kernel_size
        assert expand_ratio in self.expand_ratio

        i = self.kernel_size.index(kernel_size)*len(self.expand_ratio)
        j = self.expand_ratio.index(expand_ratio)
        x = self.choices[i+j](x, training, **kwargs)
        return x

class SuperKSMBConvBlock(Model):

    """
    "kernel size select ConvBlock
    """
    def __init__(self, 
                kernel_size,
                **kwargs):
        super(SuperKSMBConvBlock, self).__init__(name=kwargs['name'])
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size

        #self.choices = [ [[]] * len(expand_ratio) ] * len(kernel_size)
        self.choices = [] 
        for k_size in kernel_size:
            self.choices.append( SuperMBConvBlock(max_kernel_size=k_size, **kwargs) )
        
    def call(self, x, training, kernel_size=None, **kwargs):
        
        assert kernel_size in self.kernel_size
        assert expand_ratio in self.expand_ratio

        i = self.kernel_size.index(kernel_size)
        x = self.choices[i](x, training, **kwargs)
        return x

class SuperESMBConvBlock(Model):

    """
    "expand ratio select ConvBlock
    """
    def __init__(self, 
                expand_ratio, 
                **kwargs):
        super(SuperESMBConvBlock, self).__init__(name=kwargs['name'])
        self.expand_ratio = expand_ratio

        self.choices = [] 
        for ex_ratio in expand_ratio:
            self.choices.append( SuperMBConvBlock(max_expand_ratio=ex_ratio, **kwargs) )
        
    def call(self, x, training,  expand_ratio=None,**kwargs):
        
        assert expand_ratio in self.expand_ratio

        i = self.expand_ratio.index(expand_ratio)
        x = self.choices[i](x, training, **kwargs)
        return x