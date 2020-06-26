from oneshot_nas_blocks import SuperBatchNormalization
import logging
import tensorflow as tf
from supernet_train import Trainer
from utils.calculate_flops_params import get_flops_params
from oneshot_nas_net import archs_choice_with_constant
from utils.data import get_webface, get_cifar10
from oneshot_nas_net import *
from supernet import *
from random import random, choice
from copy import deepcopy
import heapq, math

class Searcher(object):
    """ Search the best arch"""
    def __init__(self, model, data, **kwargs):
        super(Searcher, self).__init__()
        #prefetch imgs
        update_bn_imgs = kwargs.get('update_bn_imgs', 30000)
        self.batch_size = kwargs.get('batch_size', 128)
        data['train_ds'] = data['train_ds'].take(update_bn_imgs).batch(self.batch_size).prefetch(200)
        data['train_num'] = update_bn_imgs
        data['val_ds'] = data['val_ds'].batch(self.batch_size).prefetch(200)
        
        
        self.data = data
        self.model = model

        self.flops_constant = kwargs.get('flops_constant', math.inf)
        self.params_constant = kwargs.get('params_constant', math.inf)
        

    def set_update_bn(layer, inference_update_stat):
        if hasattr(layer, 'layers'):
            for ll in layer.layers:
                Searcher.set_update_bn(ll, inference_update_stat)
        else:
            if isinstance(layer, (SuperBatchNormalization, tf.keras.layers.BatchNormalization)):
                if isinstance(layer, SuperBatchNormalization):
                    layer.inference_update_stat = inference_update_stat
                    logging.debug('set layer bn: %s' % str(layer))

                if inference_update_stat == True:
                    for weight in layer.weights:
                        if 'moving_var' in weight.name:
                            weight.assign(tf.ones(weight.shape))
                            logging.debug('set moving var ones: %s' % str(layer))
                        elif 'moving_mean' in weight.name:
                            weight.assign(tf.zeros(weight.shape))
                            logging.debug('set moving mean zeros: %s' % str(layer))

    def update_bn(self, arch):
        Searcher.set_update_bn(self.model, inference_update_stat=True)
        probar = tf.keras.utils.Progbar(ceil(self.data['train_num']/self.batch_size))
        #print('update bn...')
        for idx, (imgs, labels) in enumerate(self.data['train_ds']):
            self.model(imgs, True, arch)
            probar.update(idx+1)

        Searcher.set_update_bn(self.model, inference_update_stat=False)

    def get_accuracy(self, arch):
        probar = tf.keras.utils.Progbar(ceil(self.data['val_num']/self.batch_size), stateful_metrics=['val_acc'])
        acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy('val_acc')
        for idx, (imgs, labels) in enumerate(self.data['val_ds']):
            logits = self.model(imgs, False, arch)
            acc = acc_metrics(labels, logits)
            probar.update(idx+1, values=[['val_acc', acc]])
        return acc


class TopKHeap(object):
    """save the best k arch"""
    def __init__(self, k, **kwargs):
        super(TopKHeap, self).__init__()
        self.k = k
        self.data = []
        self.search_target = kwargs.get('search_target', 'acc')

    def push(self, net):
        self.data.append(net)
        if self.search_target == 'acc':
            self.data.sort(key=lambda d:d['acc'], reverse=True)
        elif self.search_target == 'flops_acc':
            self.data.sort(key=lambda d:d['acc']/d['flops_score'], reverse=True)
        else:
            raise ValueError("Unrecognized search-target: {}".format(self.search_target))

        self.data = self.data[:self.k]
        if net == self.data[0]:
            logging.info('Find a better archs, acc:{acc:.4f}, flops:{flops:.2f}, params:{params:.2f}'.format(**net))

    def get(self):
        return self.data
        

class Evolver(object):
    """docstring for Evolver"""
    def __init__(self, model, data, population_size=500, retain_length=100, random_select=0.1, mutate_chance=0.1, **kwargs):
        super(Evolver, self).__init__()

        self.population_size = population_size
        self.retain_length = retain_length
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.search_target = kwargs.get('search_target', 'acc')

        self.searcher = Searcher(model, data, **kwargs)

    def create_population(self):
        """create a population of random networks
        Return: (list): population of random archs networks
        """
        population = []
        for i in range(self.population_size):
            instance = {}

            arch, flops, params = archs_choice_with_constant(self.searcher.data['imgs_shape'], 
                                self.searcher.model.search_args, self.searcher.model.blocks_args, 
                                flops_constant=self.searcher.flops_constant, params_constant=self.searcher.params_constant, flops_params=True)
            flops_score, params_score = flops  / self.searcher.flops_constant, params  / self.searcher.params_constant
            combined_score = 0.5 * flops_score + 0.5* params_score

            logging.info("Population size + 1, total {}, with normalized score: {:.4f}, flops: {:.4f}, params: {:.4f}"
                  .format(i+1, combined_score, flops, params))
            #logging.info('flops: %.4f, params: %.4f' % (flops, params))
            instance['arch'] = arch
            instance['flops'] = flops
            instance['params'] = params
            instance['flops_score'] = flops_score
            instance['params_score'] = params_score
            instance['combined_score'] = combined_score
            population.append(instance)
        return population

    def fitness(self, arch):
        
        logging.info('start update bn...')
        self.searcher.update_bn(arch)
        logging.info('update bn done...')

        logging.info('begin cal val acc...')
        acc = self.searcher.get_accuracy(arch)
        logging.info('cal val acc done...')

        return acc

    def breed(self, mother, father):
        """makr two children
        Args:
            mother (dict): Network parameter
            father (dict): network parameter
        Return:
            (list) : Two network object
        """
        children = []
        for _ in range(2):
            child = {'arch': deepcopy(mother['arch']) }
            assert len(mother['arch']) == len(father['arch'])
            for idx, (mother_search_arg, father_search_arg) in enumerate(zip(mother['arch'], father['arch'])):
                child['arch'][idx] = child['arch'][idx]._replace(
                        width_ratio = choice([mother_search_arg.width_ratio, father_search_arg.width_ratio])\
                            if self.mutate_chance < random() else choice(self.searcher.model.search_args[idx].width_ratio),
                        expand_ratio = choice([mother_search_arg.expand_ratio, father_search_arg.expand_ratio])\
                            if self.mutate_chance < random() else choice(self.searcher.model.search_args[idx].expand_ratio),
                        kernel_size = choice([mother_search_arg.kernel_size, father_search_arg.kernel_size])\
                            if self.mutate_chance < random() else choice(self.searcher.model.search_args[idx].kernel_size),
                    )
            children.append(child)
        return children

    def evolve(self, population, topk_items, ):
        """evolve a population of network"""

        #fitness
        logging.info('start fitness...')
        for person in population:
            if 'acc' not in person.keys():
                person['acc'] = self.fitness(person['arch'])
                topk_items.push(deepcopy(person))
        
        if self.search_target == 'flops_acc': ## 
            population.sort(key=lambda x: x['acc']/x['flops_acc'], reverse=True)
        elif self.search_target == 'acc':
            population.sort(key=lambda x: x['acc'], reverse=True)
        else:
            raise ValueError('Unrecognized search target: {}'.format(self.search_target))

        # the parents we want to keep
        parents = population[:self.retain_length]

        # for those wo arenot want to keeping, we randomly keep some anyway
        for retain_person in population[self.retain_length:]:
            if self.random_select > random():
                parents.append(retain_person)

        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        logging.info('select person number {}...'.format(len(parents)))
        logging.info('start breed...')
        while len(children) < desired_length:
            mother = father = None
            while mother == father:
                mother = choice(parents)
                father = choice(parents)

            childs = self.breed(mother, father)

            for child in childs:
                if len(children) >= desired_length:
                    break

                flops, params = get_flops_params(self.searcher.data['imgs_shape'], 
                                    search_args=child['arch'], blocks_args=self.searcher.model.blocks_args, )
                flops_score, params_score = flops  / self.searcher.flops_constant, params  / self.searcher.params_constant
                combined_score = 0.5 * flops_score + 0.5* params_score
                if flops > self.searcher.flops_constant or params > self.searcher.params_constant:
                    logging.info('[SKIPPED] child model, flops:{flops:.1f}, params:{params:.1f}'.format(flops=flops, params=params))
                    continue


                logging.info("children size + 1, with normalized score: {}, flops: {:.1f}, params: {:.1f}"
                      .format(combined_score, flops, params))

                child['flops'] = flops
                child['params'] = params
                child['flops_score'] = flops_score
                child['params_score'] = params_score
                child['combined_score'] = combined_score
                children.append(child)

        parents.extend(children)
        logging.info('breed done, parents number {}...'.format(len(parents)))
        assert len(parents) == self.population_size, (len(parents), self.population_size)

        return parents



def random_search(model, data, search_iters=1000, **kwargs):
    searcher = Searcher(model, data, **kwargs)
    result = []
    probar = tf.keras.utils.Progbar(search_iters)
    for i in range(search_iters):
        arch, flops, params = archs_choice_with_constant(searcher.data['imgs_shape'], 
                            searcher.model.search_args, searcher.model.blocks_args, 
                            flops_constant=searcher.flops_constant, params_constant=searcher.params_constant, flops_params=True)
        flops_score, params_score = flops  / searcher.flops_constant, params  / searcher.params_constant
        combined_score = 0.5 * flops_score + 0.5* params_score
        

        logging.info("Target size + 1, with normalized score: {}".format(combined_score))
        logging.info('flops: %.4f, params: %.4f' % (flops, params))
        arch_list = [list(i) for i in arch]
        logging.info('archs: %s' % str(arch_list))

        logging.info('begin update bn...')
        searcher.update_bn(arch)
        logging.info('update bn done...')
    
        logging.info('begin cal val acc...')
        val_acc = searcher.get_accuracy(arch)
        logging.info('cal val acc done...')
                
        result.append( dict(flops=flops, params=params, flops_score=flops_score, params_score=params_score,
                            combined_score=combined_score, val_acc=val_acc, arch = arch_list) )

        if i % 10 == 0:
            np.save('arch_search.npy', result, allow_pickle=True)
            logging.info('save arch number: %d'% len(result))
        probar.add(1)

    return result

def genetic_search(model, data, search_iters=1000, **kwargs):
    evolver = Evolver(model, data, **kwargs)
    file_name = kwargs.get('file_name', 'genitic_search_result.npy')
    population = evolver.create_population()

    result = TopKHeap(100, search_target='acc')
    probar = tf.keras.utils.Progbar(search_iters)
    logging.info('start evolve...')
    for _ in range(search_iters):
        probar.add(1)
        population = evolver.evolve(population, topk_items=result)
        np.save(file_name, result.get(), allow_pickle=True)
    logging.info('evolve done...')
    return result.get()



def search(type='genetic_search'):
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    type_name = 'mobilenetv2-b0'
    load_path = 'training_data/checkpoing/weights_073-5.1185-0.2295.tf/'
    file_name = 'archs_search.npy'
    model = get_nas_model(type_name, blocks_type='bes', load_path=load_path)

    data = get_webface()

    flops_constant = 120
    params_constant = math.inf

    logging.info('random search begining...')

    if type == 'random_search':
        result = random_search(model, data, search_iters=1000, flops_constant=flops_constant, params_constant=params_constant)
    else:
        result = genetic_search(model, data, search_iters=10, 
                flops_constant=flops_constant, params_constant=params_constant, search_target='acc', retain_length=20, population_size=100, batch_size=200,
                                    file_name=file_name)

    np.save(file_name, result, allow_pickle=True)
    logging.info('search done')

if __name__ == '__main__':
    import os,logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    search('genetic_search')
