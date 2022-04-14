import os
import json
import numpy as np 
import pickle as pkl
import bcolz
import torch
from nltk.corpus import wordnet

replace_dict = {
    'aquarium_fish': 'fish',
    'maple_tree': 'maple',
}

replace_dict_glove = {
    "lawn_mower": "mower",
    "oak_tree": "oak",
    "palm_tree": "palm",
    "pickup_truck": "pickup",
    "pine_tree": "pine",
    "sweet_pepper": "papper",
    "willow_tree": "willow",
    "Angora": "Angora_rabbit",
    "Lhasa": "Lhasa_apso",
    "Chihuahua": "dog"
}

replace_dict_fastext = {
    "aquarium_fish": "aquarium",
    "lawn_mower": "lawn",
    "maple_tree": "maple",
    "oak_tree": "oak",
    "palm_tree": "palm",
    "pickup_truck": "truck",
    "pine_tree": "pine",
    "sweet_pepper": "pepper",
    "willow_tree": "willow"
}

def get_label_names(dataset='cifar100'):
    if dataset == 'cifar100':
        with open('./cifar100_origin/cifar100_origin/cifar-100-python/meta', 'rb') as f:
            ret = pkl.load(f)['fine_label_names']
        return ret

    if dataset == 'cifar10':
        l = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return l

    if dataset == 'imagenet':
        ret = []
        with open('imagenet_class_index.json', 'r') as f:
            d = json.load(f)
            for key, val in d.items():
                ret.append(val[1])
            assert (len(ret) == 1000)
        return ret

    if dataset == 'timagenet':
        id_name_dict = {}
        with open('/home/e/e0517241/datasets/tiny-imagenet-200/words.txt', 'r') as f:
            for line in f.readlines():
                ls = line.split('\t')
                name = ls[1][:-1]
                if ',' in name:
                    name = name.split(',')[0]
                if ' ' in name:
                    name = '_'.join(name.split(' '))
                id_name_dict[ls[0]] = name 
        
        wnids = [] 
        with open('/home/e/e0517241/datasets/tiny-imagenet-200/wnids.txt', 'r') as f:
            for line in f.readlines():
                wnids.append(line[:-1])
        
        ret = []
        for id in wnids:
            ret.append(id_name_dict[id].lower())

        return ret


def subsitite(d, replace_dict):
    for i,k in enumerate(d):
        if k in replace_dict:
            d[i] = replace_dict[k]
    return d

def get_similarity_matrix(dataset="cifar100", source="wordnet"):
    if source == "bert":
        print("load from bert")
        t = np.load("bert.npy")
        for i in range(100):
            t[i][i] = 1.
        return t
    if source == "gpt2":
        print("load from gpt2")
        t = np.load("gpt2.npy")
        for i in range(100):
            t[i][i] = 1.
        return t
    file_name = '{}_similarity_matrix_{}.npy'.format(dataset, source)
    if os.path.exists(file_name):
        print("returning sim mat: ", file_name)
        return np.load(file_name)

def gen_similarity_matrix_wordnet(label_list=None, dataset='cifar100'):
    file_name = '{}_similarity_matrix_wordnet.npy'.format(dataset)

    n = len(label_list)
    similarity = np.eye(n)
    concepts = []
    for i in range(n):
        concepts.append(wordnet.synset(label_list[i]+'.n.01'))
    for i in range(n):
        for j in range(i+1, n):
            similarity[i][j] = wordnet.path_similarity(concepts[i], concepts[j])
            similarity[j][i] = similarity[i][j]
    print("!!!similarity matrix saved!")
    np.save(file_name, similarity)
    return similarity


def load_fastext_vectors(fname):
    import io
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(tokens[1:])
    return data


def gen_similarity_matrix_fastext(label_list=None, dataset="cifar100"):
    file_name = '{}_similarity_matrix_fastext.npy'.format(dataset)
    vectors = load_fastext_vectors("./fastext/wiki-news-300d-1M.vec")

    wordembeddings = np.zeros((100, 300))
    similarity = np.zeros((100, 100))
    for i, label in enumerate(label_list):
        if label in vectors:
            # print(vectors[label].shape)
            # print(np.array(vectors[label]))
            wordembeddings[i, :] = vectors[label]
        # else:
            # print("not in: ", i, label)

    n = 100
    for i in range(n):
        for j in range(i, n):
            a = wordembeddings[i]
            b = wordembeddings[j]
            if a is None or b is None:
                similarity[i][j] = similarity[j][i] = None
            else:
                t = (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))
                similarity[i][j] = ( t + 1 ) / 2.
                similarity[j][i] = similarity[i][j]

    np.save(file_name, similarity)
    return similarity


def load_glove_as_dict(path: str, dimension: int = 50, identifier='42B'):
    vectors = bcolz.open('{}/glove.{}.{}d.dat'.format(path, identifier, dimension))[:]
    words = pkl.load(open('{}/glove.{}.{}d.50_words.pkl'.format(path, identifier, dimension), 'rb'))
    word2idx = pkl.load(open('{}/glove.{}.{}d.50_idx.pkl'.format(path, identifier, dimension), 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove

def gen_similarity_matrix_glove(label_list=None, dataset='cifar100'):
    file_name = '{}_similarity_matrix_glove.npy'.format(dataset)
    # label_list = subsitite(label_list, {**replace_dict_glove, **replace_dict})
    n = len(label_list)
    similarity = np.zeros((n, n))
    
    glove_path = '/home/e/e0517241/anasyn/AnaSyn/data/glove'
    identifier = '42B'
    print("start loading glove embeddings!")
    glove_vec = load_glove_as_dict(glove_path, 300, identifier)
    print("loadding glove embedding done!")
    # if dataset == "imagenet":
        # for i, label in enumerate(label_list):
            # if '_' in label:
                # label_list[i] = label.split('_')[-1]
    for i, label in enumerate(label_list):
        # label_synset = wordnet.synset(label + ".n.01")
        print_list = []
        while (label not in glove_vec.keys()) and label is not None:
            word_synset = wordnet.synset(label + ".n.01")
            synset = word_synset.hypernyms()
            if len(synset) == 0:
                print("[!]", label, "has no hypernym")
                label = None
            else:
                print_list.append(label)
                long_name = str(synset[0])
                start = 8
                end = long_name.index('.')
                label_name = long_name[start:end]
                # print(label, "->", label_name)
                label = label_name
        label_list[i] = label
        print_list.append(label)
        if len(print_list) > 1:
            print(' -> '.join(print_list))

    # exit(0)
    wordembeddings = []
    for i in range(n):
        if label_list[i] is None:
            wordembeddings.append(None)
        else:
            wordembeddings.append(glove_vec[label_list[i]])

    for i in range(n):
        for j in range(i, n):
            a = wordembeddings[i]
            b = wordembeddings[j]
            if a is None or b is None:
                similarity[i][j] = similarity[j][i] = None
            else:
                t = (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))
                similarity[i][j] = ( t + 1 ) / 2.
                similarity[j][i] = similarity[i][j]
    print(similarity[:10, :10])
    np.save(file_name, similarity)
    return similarity


def visualize(label_list, dataset, similarity, save_dir='sim_vis', fmt='png'):
    import seaborn as sb; sb.set_theme()
    import matplotlib.pyplot as plt
    if len(label_list) == 200:
        plt.figure(figsize=(40, 40))
    elif len(label_list) == 1000:
        plt.figure(figsize=(200, 200))
    else:
        plt.figure(figsize=(30, 30))

    name = '{}_similarity_matrix_{}'.format(dataset, similarity)
    matrix = np.load('{}.npy'.format(name))

    img = sb.heatmap(matrix, xticklabels=label_list, yticklabels=label_list)
    fig = img.get_figure()
    fig.savefig('{}/{}.{}'.format(save_dir, name, fmt))
    print("saved: " ,dataset, similarity)


def find_difference(dataset, similarities, label_names):
    mat1 = np.load('{}_similarity_matrix_{}.npy'.format(dataset, similarities[0]))
    mat2 = np.load('{}_similarity_matrix_{}.npy'.format(dataset, similarities[1]))
    n_class = mat1.shape[0]

    i = 0
    print(n_class)
    nconflict = 0
    for l1 in range(n_class):
        for l2 in range(l1+1, n_class):
            for l3 in range(l2+1, n_class):
                # print(l1, l2, l3)
                # continue99 *

                i += 1
                if i % 100 == 0:    
                    print("count {}".format(i))
                    print("nconflict{}".format(nconflict))
                    print()

                # l1, l2, l3 = np.random.choice(list(range(n_class)), 3)
                l = [l1, l2, l3]
                
                d1_12 = mat1[l1, l2]
                d1_23 = mat1[l2, l3]
                d1_31 = mat1[l1, l3]
                ls1 = [d1_12, d1_23, d1_31]
                idx1 = sorted([0, 1, 2], key=lambda i:ls1[i])

                d2_12 = mat2[l1, l2]
                d2_23 = mat2[l2, l3]
                d2_31 = mat2[l1, l3]
                ls2 = [d2_12, d2_23, d2_31]
                idx2 = sorted([0, 1, 2], key=lambda i:ls2[i])

                if not (idx1 == idx2):
                    nconflict += 1
                    if d1_12 > 0.1 and d1_23 > 0.1 and d1_31 > 0.1:
                        print(nconflict, "/", i, label_names[l1], label_names[l2], label_names[l3])
                        print(d1_12, d1_23, d1_31)
                        print(d2_12, d2_23, d2_31)


if __name__ == '__main__':
    ds = ['cifar100']
    sims = ['wordnet', "glove"]
    
    label_list = get_label_names("cifar100")
    label_list = subsitite(label_list, replace_dict_fastext)
    gen_similarity_matrix_fastext(label_list)
    # for d in ds:
        # LL = get_label_names(dataset=d)
        # for i, n in enumerate(LL):
            # print(i, n)
        # exit(0)
        # find_difference(ds[0], sims, LL)
    # LL = subsitite(get_label_names())
    # r = gen_similarity_matrix_glove(LL, dataset='timagenet')
    # print(r[:5, :5])