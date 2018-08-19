import pickle
import numpy as np

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


def load_all_images(cifar10_dataset_folder_path):
    all_batches = load_cfar10_batch(cifar10_dataset_folder_path, 1)[0]
    for i in range(2,7):
        all_batches = np.concatenate((all_batches, load_cfar10_batch(cifar10_dataset_folder_path, i)[0]))
    return all_batches

def normalize(x):

    return x/255.0


def normalized_generated(x):
    maxi = np.max(x)
    mini = np.min(x)
    return (x-mini)/(maxi-mini)



def normalized_data(images):
    normalized_images = []
    for i in range(len(images)):
        normalized_images.append(normalize(images[i]))
    return np.array(normalized_images)

def next_batch(images, batch_size):
    idx = np.random.randint(0, len(images), (batch_size))
    return images[idx,:]
