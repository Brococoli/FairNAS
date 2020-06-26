import tensorflow as tf
from tensorflow.keras import Model
from collections import namedtuple
from math import ceil
from random import choice
from oneshot_nas_blocks import *
import logging
import numpy as np
from utils.calculate_flops_params import get_flops_params
from copy import deepcopy




BlockArgs = namedtuple('BlockArgs',
                       [ 'kernel_size',
                         'num_repeat',
                         'channels',
                         'expand_ratio',
                         'id_skip',
                         'strides',
                         'se_ratio',
                         ])

SearchArgs = namedtuple('SearchArgs',
                       [ 'width_ratio',
                         'kernel_size',
                         'expand_ratio',
                         ])

DEFAULT_BLOCKS_ARGS = [
        BlockArgs(kernel_size=3, num_repeat=1, channels=16,
                  expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=2, channels=24,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=32,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=4, channels=64,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=96,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, channels=160,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=1, channels=320,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
        ]

SPOS_BLOCKS_ARGS = [
        BlockArgs(kernel_size=3, num_repeat=4, channels=48,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=4, channels=160,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=8, channels=320,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=4, channels=640,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
]


DEFAULT_SEARCH_ARGS = [
SearchArgs(width_ratio=[1], kernel_size=[3], expand_ratio=[6]) for i in range(sum(arg.num_repeat for arg in DEFAULT_BLOCKS_ARGS))
]

SPOS_SEARCH_ARGS = [
SearchArgs(width_ratio=np.arange(0.2,2.1,0.2), kernel_size=[3,5,7], expand_ratio=range(2,7) if i else [1]) for i in range(sum(arg.num_repeat for arg in SPOS_BLOCKS_ARGS))
]


def archs_choice(search_args):
  archs = deepcopy(search_args)
  for i, arg in enumerate(search_args):
    archs[i] = archs[i]._replace(width_ratio=choice(arg.width_ratio),
                  kernel_size=choice(arg.kernel_size),
                  expand_ratio=choice(arg.expand_ratio))
  return archs

def archs_choice_with_constant(input_shape, search_args, blocks_args, flops_constant, params_constant, flops_params=False):
  while True:
    arch = archs_choice(search_args)
    flops, params = get_flops_params(input_shape, search_args=arch, blocks_args=blocks_args)
    if flops < flops_constant and params < params_constant:
      break
    #logging.debug('Search a arch: %d %d' % (flops, params))
  if flops_params == False:
    return arch
  else:
    return arch, flops, params



def make_divisible(v, divisor, min_value = None):
    """ 
    It ensure that all layers have a channel number that is divisible by 8
    It can be seen here:
            https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v+=divisor
    return int(new_v)

class SinglePathOneShot(Model):
    def __init__(self, 
                dropout_rate=0.2,
                drop_connect_rate=0.,
                depth_divisor=8,
                search_args=DEFAULT_SEARCH_ARGS,
                blocks_args=DEFAULT_BLOCKS_ARGS,
                model_name='SinglePathOneShot',
                activation='relu6',
                use_se=True,
                include_top=True,
                pooling=None,
                initializer = 'he_normal',
                num_classes=None,
                blocks_type='mix',
                weight_decay = 1e-4, **kwargs):
        super(SinglePathOneShot, self).__init__(name=model_name, **kwargs)

        self.search_args = search_args
        self.blocks_args = blocks_args


        self.round_width = lambda x : make_divisible(x, depth_divisor)


        self.stem = tf.keras.Sequential([layers.Conv2D(32, 
            kernel_size=3, 
            strides=2,
            use_bias=False, 
            padding='SAME', 
            kernel_initializer=initializer, 
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            ),
            SuperBatchNormalization(32, momentum=0.9, epsilon=1e-5,),
            Activation(activation)], name='stem')     

        block_dropout_rates = tf.linspace(0.0, drop_connect_rate,
                sum(arg.num_repeat for arg in blocks_args))

        num_blocks = 0
        max_filters_in = 32
        self.blocks = []

        self.segments_idx = [0]*len(search_args)
        assert len(search_args) == sum(arg.num_repeat for arg in blocks_args), (len(search_args) , sum(arg.num_repeat for arg in blocks_args))

        for idx, block_arg in enumerate(blocks_args):
            num_repeat = block_arg.num_repeat
            channels = block_arg.channels
            id_skip = block_arg.id_skip
            strides = block_arg.strides
            se_ratio = block_arg.se_ratio if use_se else None


            for i in range(num_repeat):
                expand_ratio = search_args[num_blocks].expand_ratio
                max_expand_ratio = max(expand_ratio)
                kernel_size = search_args[num_blocks].kernel_size
                max_kernel_size = max(kernel_size)
                max_filters_out = self.round_width(max(search_args[num_blocks].width_ratio)*channels)
                max_filters_out = max(max_filters_out, 16)
                logging.debug('block args: %d, %d, %d' %(max_expand_ratio, max_kernel_size, max_filters_out))
                if blocks_type == 'mix': # single path archs
                    self.blocks.append(
                        SuperMBConvBlock(max_filters_in=max_filters_in,  
                            max_filters_out=max_filters_out,  
                            max_expand_ratio=max_expand_ratio,
                            max_kernel_size = max_kernel_size, 
                            se_ratio=se_ratio, 
                            strides=strides, 
                            weight_decay=weight_decay, 
                            use_shortcut=id_skip, 
                            drop_connect_rate=block_dropout_rates[num_blocks],
                            activation=activation,  
                            name='block_%d_%d' % (idx, i))
                    )

                elif blocks_type == 'ks': # kernel size choice blocks 
                    self.blocks.append(
                        SuperKSMBConvBlock(max_filters_in=max_filters_in,  
                            max_filters_out=max_filters_out,  
                            max_expand_ratio=max_expand_ratio,
                            kernel_size = kernel_size, 
                            se_ratio=se_ratio, 
                            strides=strides, 
                            weight_decay=weight_decay, 
                            use_shortcut=id_skip, 
                            drop_connect_rate=block_dropout_rates[num_blocks],
                            activation=activation,  
                            name='block_%d_%d' % (idx, i))
                    )
                elif blocks_type == 'es': # expand ratio select blocks 
                    self.blocks.append(
                        SuperESMBConvBlock(max_filters_in=max_filters_in,  
                            max_filters_out=max_filters_out,  
                            expand_ratio=expand_ratio,
                            max_kernel_size = max_kernel_size, 
                            se_ratio=se_ratio, 
                            strides=strides, 
                            weight_decay=weight_decay, 
                            use_shortcut=id_skip, 
                            drop_connect_rate=block_dropout_rates[num_blocks],
                            activation=activation,  
                            name='block_%d_%d' % (idx, i))
                    )
                elif blocks_type == 'kes': # kernel and expand_ratio select blocks 
                    self.blocks.append(
                        SuperKESMBConvBlock(max_filters_in=max_filters_in,  
                            max_filters_out=max_filters_out,  
                            expand_ratio=expand_ratio,
                            kernel_size = kernel_size, 
                            se_ratio=se_ratio, 
                            strides=strides, 
                            weight_decay=weight_decay, 
                            use_shortcut=id_skip, 
                            drop_connect_rate=block_dropout_rates[num_blocks],
                            activation=activation,  
                            name='block_%d_%d' % (idx, i))
                    )
                else:
                    raise ValueError('blocks type error, no ' + str(blocks_type))

                strides = [1,1]
                max_filters_in = max_filters_out
                self.segments_idx[num_blocks] = idx
                num_blocks += 1


        print('num_blocks:', num_blocks)
        self.num_blocks = num_blocks
        
        assert num_blocks == len(search_args)
        

        self.head = tf.keras.Sequential([
                            SuperConv2d(max_filters_in=max_filters_out, 
                                 max_filters_out=1280,
                                 max_kernel_size=1, 
                                 strides=1, 
                                 padding='SAME', 
                                 use_bias = False,
                                 kernel_initializer=initializer, 
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 name='sconv',
                                 ),
                            SuperBatchNormalization(1280, momentum=0.9, epsilon=1e-5,),
                            Activation(activation)
                    ], name='head')
        
        self.top = tf.keras.Sequential([], name='top')
        if include_top:
            self.top.add( layers.GlobalAvgPool2D(name = 'avgpool') )

            if dropout_rate and ( 0.0 < dropout_rate < 1.0):
                self.top.add( layers.Dropout(dropout_rate, name='dp') )
                
            self.top.add( layers.Dense(num_classes, 
                                     kernel_initializer='he_normal', 
                                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay)) )

        else:
            if pooling == 'avg':
                self.top.add( layers.GlobalAvgPool2D(name = 'avgpool') )
            elif pooling == 'max':
                self.top.add( layers.GlobalMaxPool2D(name = 'maxpool') )
            elif not pooling or pooling == '':
                pass
            else:
                raise ValueError('no pooling:', pooling)
        
    
    def choice_architecture(self):
        arch = archs_choice(self.search_args)
        return arch


    def call(self, x, training, search_args=None):
        #logger = logging.getLogger('supernet')

        search_args = self.choice_architecture() if search_args == None else search_args
        logging.debug('x.shape: %s' % str(x.shape))
        x = self.stem(x, training)
        logging.debug('x.shape: %s' % str(x.shape))

        for idx, block in enumerate(self.blocks):
            width_ratio, kernel_size, expand_ratio = search_args[idx]
            filters_out = self.round_width(width_ratio*self.blocks_args[self.segments_idx[idx]].channels)
            filters_out = max(filters_out, 16)

            #logging.debug('search args: {}, {}, {}'.format(*search_args[idx]))
            x = block(x, training, expand_ratio=expand_ratio, 
                               kernel_size=kernel_size, filters_out=filters_out) 

            logging.debug('x.shape: %s' % str(x.shape))   
        
        x = self.head(x, training)
        logging.debug('x.shape: %s' % str(x.shape))
        x = self.top(x, training)
        logging.debug('x.shape: %s' % str(x.shape))
        
        return x


if __name__ == '__main__':
    from supernet import get_nas_model
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    model = get_nas_model('spos-b0')

    t = tf.random.normal((1,112,96,3))
    model(t, True)
    logging.debug('')
    model(t, True)
    #print()
    logging.debug('')
    model(t, True)

    arch = archs_choice(SPOS_SEARCH_ARGS)
    logging.debug('')
    model(t, True, arch)
    logging.debug('')
    model(t, True, arch)
