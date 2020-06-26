import pathlib
import random
import numpy as np
import tensorflow as tf

def get_webface(path='/home/wonder/lab/dataset/WebFace', verbose=True):
    data = {}
    data_root = pathlib.Path(path + '/set')
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    def get_data(path, data_type):
        with open(path + '/' + data_type + '_' + 'set.txt') as f:
            l = f.read()
        all_image_paths = l.split('\n')
        all_image_paths = [path + '/set/' + p for p in all_image_paths if p != '']

        random.shuffle(all_image_paths)
        
        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

        
        def load_image(path):
            img_raw = tf.io.read_file(path)
            img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
            img_tensor = tf.cast(img_tensor, tf.float32)
            return img_tensor

        def image_augment(img_tensor):
            img_tensor = img_tensor / 255.0
            img_tensor = tf.image.per_image_standardization(img_tensor)
            img_tensor =  tf.image.random_flip_left_right(img_tensor)
            return img_tensor
        
        if data_type == 'train':
            data['train_num'] = len(all_image_paths)
        if data_type == 'val':
            data['val_num'] = len(all_image_paths)
        if data_type == 'test':
            data['test_num'] = len(all_image_paths)
        
        path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
        image_ds = path_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
                          batch(256).\
                          map(image_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
        
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int16))
        #label_ds = label_ds.map(squeeze)
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        return image_label_ds

    data['train_ds'] = get_data(path, 'train')
    data['imgs_shape'] = (112,96,3)
    data['val_ds'] = get_data(path, 'val')
    data['num_classes'] = 10575
    if verbose:
        sample = next(iter(data['train_ds'].take(1)))
        print('train images shape:', sample[0].shape, ', images shape:', sample[1].shape, ', len', data['train_num'])
        
        sample = next(iter(data['val_ds'].take(1)))
        print('val images shape:', sample[0].shape, ', images shape:', sample[1].shape, ', len', data['val_num'])
        
    return data

def get_cifar10(verbose=True):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = tf.cast(X_train, tf.float32) / 255.0
    X_test = tf.cast(X_test, tf.float32) / 255.0
    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    X_train, X_test = tf.image.per_image_standardization(X_train), tf.image.per_image_standardization(X_test)

    X_val = X_train[-10000:]
    y_val = y_train[-10000:]
    X_train = X_train[:-10000]
    y_train = y_train[:-10000]

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(1000)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(1000)
    # train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    # test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    data = {     'train_ds': train_ds, 
                 'val_ds' : val_ds, 
                 'test_ds' : test_ds, 
                 'train_num' : X_train.shape[0],
                 'val_num' : X_val.shape[0],
                 'test_num' : X_test.shape[0],
                 'imgs_shape':(32,32,3),
                 'num_classes':10, 
            }

    if verbose:
        for i,j in data.items():
            if type(j) != int:
                print(i, j, len(list(j)))
    return data