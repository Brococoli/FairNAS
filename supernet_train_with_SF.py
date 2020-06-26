from supernet import *
import tensorflow as tf
import logging
from oneshot_nas_net import archs_choice
from utils.calculate_flops_params import get_flops_params
from utils.data import get_webface, get_cifar10
import math
from math import ceil
from copy import deepcopy
import random, itertools, pdb

class Trainer(object):
    """supernet trainer"""
    def __init__(self, model, data, optimizer, **kwargs):
        super(Trainer, self).__init__()
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.flops_constant = kwargs.get('flops_constant', math.inf)
        self.params_constant = kwargs.get('params_constant', math.inf)

        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')


    def lr_plan(self, epoch):
        plan = [1e-3, 1e-4, 1e-5]
        lr = plan[min(len(plan)-1, epoch//45)]
        print('learning rate:', lr)
        tf.keras.backend.set_value(self.optimizer.lr, lr)

    def search_plan(self, epoch):
        return self.model.search_args
        
    def train_step(self, images, labels, archs_args):
        losses = []
        with tf.GradientTape() as g:
            for archs_arg in archs_args:
                logits = self.model(images, True, search_args=archs_arg)
                loss = self.loss_func(y_true = labels, y_pred = logits)
                loss += sum(self.model.losses)
                losses.append(loss)

        #pdb.set_trace()
        grads = g.gradient(losses, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        acc = self.train_acc(y_true = labels, y_pred = logits)
        loss = self.train_loss(loss)
        return loss, acc

    def val_step(self, images, labels, search_args):
        logits = self.model(images, False, search_args=search_args)
        loss = self.loss_func(y_true = labels, y_pred = logits)
        loss += sum(self.model.losses)

        acc = self.val_acc(y_true = labels, y_pred = logits)
        loss = self.val_loss(loss)
        return loss, acc

    def get_archs(self,epoch):
        search_args = self.search_plan(epoch)
        model_permutation = []
        layers_permulation = []
        seg_idx = -1
        for idx, search_arg in enumerate(search_args):
            
            if self.model.segments_idx[idx] != seg_idx:
                width_ratio = list(search_arg.width_ratio) 
                random.shuffle(width_ratio)
                seg_idx = self.model.segments_idx[idx]
            #pdb.set_trace()
            kernel_size = list(search_arg.kernel_size)
            random.shuffle(kernel_size)
            expand_ratio = list(search_arg.expand_ratio)
            random.shuffle(expand_ratio)
            #print('width_ratio', width_ratio)
            
            layers_permulation.append(list(itertools.product(width_ratio, kernel_size, expand_ratio)))
        #print('layer_choice', layers_permulation)
        #print(layers_permulation)
        for idx in range(len(layers_permulation[0])):
            archs_args = []
            for archs_arg in layers_permulation:
                width_ratio, kernel_size, expand_ratio = archs_arg[idx]
                archs_args.append(SearchArgs(width_ratio=width_ratio, kernel_size=kernel_size, expand_ratio=expand_ratio))
            model_permutation.append(archs_args)
        return model_permutation

    def train(self, epochs, batch_size=128):
        train_ds = self.data['train_ds']
        train_num = self.data['train_num']
        val_ds = self.data['val_ds']
        val_num = self.data['val_num']
        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(200)
        val_ds = val_ds.batch(batch_size).prefetch(200)

        epochs_probar = tf.keras.utils.Progbar(epochs)   
        for epoch in range(epochs):
            train_probar = tf.keras.utils.Progbar(ceil(train_num/batch_size), stateful_metrics=['accuracy', 'loss'])

            self.lr_plan(epoch)
            epochs_probar.update(epoch)
            #print()
            for idx, (images, labels) in enumerate(train_ds):
                archs_args = self.get_archs(epoch) 
                #pdb.set_trace()
                loss, acc = self.train_step(images, labels, archs_args)
                train_probar.update(idx+1, values=[['accuracy', acc], ['loss', loss]])
            """
            val_probar = tf.keras.utils.Progbar(ceil(val_num/batch_size), stateful_metrics=['val_accuracy', 'val_loss'])
            for idx, (images, labels) in enumerate(val_ds):
                archs = self.get_archs(epoch) 
                loss, acc = self.val_step(images, labels, archs)
                val_probar.update(idx+1, values=[['val_accuracy', acc], ['val_loss', loss]])
            """

            self.model.save_weights('training_data/checkpoing/'+\
                'weights_{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.tf/'.format(epoch=epoch, val_loss=loss, val_accuracy=acc))
            logging.info('save the weights..')

            self.train_acc.reset_states()
            self.val_acc.reset_states()
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            
            


def train():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


    logging.info('beging train...')

    model = get_nas_model('mobilenetv2-fairnas', blocks_type='mix', load_path='', num_classes=10)
    logging.debug('get a nas model')

    data = get_cifar10()
    
    #opt = tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.9, nesterov=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    trainer = Trainer(model, data, optimizer=opt, flops_constant=100, params_constant=math.inf, )
    logging.debug('get a trainer')



    trainer.train(90, 128)


if __name__ == '__main__':
    import os,logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #tf.get_logger().setLevel(logging.ERROR)
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    train()
