"""
@File   :  utils.py
@Time   :  2022/05/6
@Author  :  Wu Yanan
@Contact :  yanan.wu@bupt.edu.cn
"""
from tkinter.messagebox import NO
from types import new_class
from typing import List
import os
import json
import pandas as pd
import itertools
import matplotlib
from sklearn.metrics import confusion_matrix
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import set_session
import numpy as np
import random as rn
from sklearn.decomposition import PCA
import torch

# SEED = 123
# tf.random.set_random_seed(SEED)
def setup_seed(SEED):
    np.random.seed(SEED)
    rn.seed(SEED)
    tf.set_random_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)

def naive_arg_topK(matrix, K, axis=1): # 对行元素从小到到排序，返回索引
    full_sort = np.argsort(matrix, axis=axis)
    return full_sort.take(np.arange(K), axis=axis)


def set_allow_growth(device="1"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.visible_device_list = device
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def load_data(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train', 'valid', 'test']:
        with open("./data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("./data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_errordata(errorpath):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['error']:
        with open(errorpath + "/error.seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open(errorpath + "/error.label") as fp:
            labels.extend(fp.read().splitlines())
    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_errorpreddata(errorpath):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['error']:
        with open(errorpath + "/error.seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open(errorpath + "/error.predlabel") as fp:
            labels.extend(fp.read().splitlines())
    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_traindata(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['train']:
        with open("./data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("./data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def load_testdata(dataset):
    texts = []
    labels = []
    partition_to_n_row = {}
    for partition in ['test']:
        with open("./data/" + dataset + "/" + partition + ".seq.in") as fp:
            lines = fp.read().splitlines()
            texts.extend(lines)
            partition_to_n_row[partition] = len(lines)
        with open("./data/" + dataset + "/" + partition + ".label") as fp:
            labels.extend(fp.read().splitlines())

    df = pd.DataFrame([texts, labels]).T
    df.columns = ['text', 'label']
    return df, partition_to_n_row

def compute_kl(probs):
    samples_num = probs.size()[0]
    labels_num = probs.size()[1]
    uniform = np.ones((samples_num,labels_num),float)/labels_num
    kl = np.sum(probs * np.log(probs/uniform),axis=1)

    print("(kl): Mean = {mean}, Var = {var}, Max = {max}, Min = {min}, Median = {median}".format(
                mean = kl.mean(),var = kl.var(),max=kl.max(),min=kl.min() ,median=np.median(kl)))
    return kl


def compute_energy_score(prob,T):
    '''
    Params:
        - logits, (batchsize,cluster_num)
    Returns:
        - energy_score, (batchsize,1)
    '''
    to_np = lambda x: x.data.cpu().numpy()
    prob = torch.from_numpy(prob)
    energy_score = -to_np((T*torch.logsumexp(prob /  T, dim=1)))
    return energy_score

def construct_confused_pairs(true_labels,pred_labels,all_labels):
    """ Construct confused label pairs set,
        In order to avoid too many confused pairs, choose at most one per class.
    @ Input:
        
    @ Return:
    """
    error_cm = confusion_matrix(true_labels,pred_labels,all_labels)
    # x_idx,y_idx = np.nonzero(error_cm)
    # error_pairs = []
    # for i in range(x_idx.shape[0]):
    #     error_pairs.append((all_labels[x_idx[i]],all_labels[y_idx[i]]))
    # print("error_pairs = ",len(error_pairs),error_pairs)
    error_pairs = []
    max_error_index = np.argmax(error_cm, axis=1)
    for i in range(error_cm.shape[0]):
        error_pairs.append((all_labels[i],all_labels[max_error_index[i]]))
    return error_pairs

def get_score(cm):
    fs = []
    ps = []
    rs = []
    n_class = cm.shape[0]
    correct = []
    total = []
    for idx in range(n_class):
        TP = cm[idx][idx]
        correct.append(TP)
        total.append(cm[idx].sum())
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)
        ps.append(p * 100)
        rs.append(r * 100)

    f = np.mean(fs).round(2)
    p = np.mean(ps).round(2)
    r = np.mean(rs).round(2)
    p_seen = np.mean(ps[:-1]).round(2)
    r_seen = np.mean(rs[:-1]).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    p_unseen = round(ps[-1], 2)
    r_unseen = round(rs[-1], 2)
    f_unseen = round(fs[-1], 2)
    acc = (sum(correct) / sum(total) * 100).round(2)
    acc_in = (sum(correct[:-1]) / sum(total[:-1]) * 100).round(2)
    acc_ood = (correct[-1] / total[-1] * 100).round(2)

    print(f"Overall(macro): , f:{f},  acc:{acc}")
    print(f"Seen(macro): , f:{f_seen}, acc:{acc_in}")
    print(f"Uneen(Experiment) , f:{f_unseen} r:{r_unseen}\n")

    return f, f_seen,  acc_in, r_seen, f_unseen,  p_unseen, r_unseen

def get_errors(cm,classes):
    """  传入混淆矩阵和类别标签，返回topk个错误标签及错误数量 """
    n_class = len(classes)
    errors = {}
    # 计算除了预测为unseen的错误外，其失误数量排序
    for idx in range(n_class):
        tp=cm[idx][idx]
        error = cm[idx].sum()-tp-cm[idx][-1] if cm[idx].sum() != 0 else 0
        errors[classes[idx]]=error
    paixu_group = sorted(errors.items(),key=lambda item:item[1],reverse=True) #[("unseen",74),()]
    top_error_classes = []
    for item in paixu_group:
        top_error_classes.append(item[0]) # ["unseen",""]
    return paixu_group,top_error_classes

def plot_confusion_matrix(output_dir, cm, classes, normalize=False,
                          title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mat.png"))


def mahalanobis_distance(x: np.ndarray,
                         y: np.ndarray,
                         covariance: np.ndarray) -> float:
    """
    Calculate the mahalanobis distance.

    Params:
        - x: the sample x, shape (num_features,)
        - y: the sample y (or the mean of the distribution), shape (num_features,)
        - covariance: the covariance of the distribution, shape (num_features, num_features)

    Returns:
        - score: the mahalanobis distance in float

    """
    num_features = x.shape[0]

    vec = x - y
    cov_inv = np.linalg.inv(covariance)
    bef_sqrt = np.matmul(np.matmul(vec.reshape(1, num_features), cov_inv), vec.reshape(num_features, 1))
    return np.sqrt(bef_sqrt).item()


def confidence(features: np.ndarray,
               means: np.ndarray,
               distance_type: str,
               cov: np.ndarray = None) -> np.ndarray:
    """
    Calculate mahalanobis or euclidean based confidence score for each class.

    Params:
        - features: shape (num_samples, num_features)
        - means: shape (num_classes, num_features)
        - cov: shape (num_features, num_features) or None (if use euclidean distance)

    Returns:
        - confidence: shape (num_samples, num_classes)
    """
    assert distance_type in ("euclidean", "mahalanobis")

    num_samples = features.shape[0]
    num_features = features.shape[1]
    num_classes = means.shape[0]
    if distance_type == "euclidean":
        cov = np.identity(num_features)

    features = features.reshape(num_samples, 1, num_features).repeat(num_classes,
                                                                     axis=1)  # (num_samples, num_classes, num_features)
    means = means.reshape(1, num_classes, num_features).repeat(num_samples,
                                                               axis=0)  # (num_samples, num_classes, num_features)
    vectors = features - means  # (num_samples, num_classes, num_features)
    cov_inv = np.linalg.inv(cov)
    bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                         vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
    result = np.sqrt(bef_sqrt)
    result[np.isnan(result)] = 1e12  # solve nan
    return result

def com_IntraVar_InterDistance(feature_train_seen,y_train_seen,prob_train_seen,feature_test_seen,prob_test_seen,feature_test,test_data):
    # Calculate intra-class variance and inter-class distance
    '''
    @ Input:

    @ Return:
    '''
    print("********************************* 计算类内方差 类间距离 *********************************")
    var_and_mean_dis(feature_train_seen, labels=y_train_seen,norm = True)
    var_and_mean_dis(feature_train_seen, labels=y_train_seen,norm = False)
    pairs = [("change_user_name","user_name"),("change_user_name","change_ai_name"),("user_name","what_is_your_name"),("change_ai_name","what_is_your_name"),("redeem_rewards","rewards_balance"),("ingredients_list","recipe"),("shopping_list","shopping_list_update"),("play_music","change_speed"),("payday","change_user_name"),("distance","directions")]
    for pair in pairs:
        print("\n")
        cm_pair = list(pair)
        cm_pair_index = y_train_seen[y_train_seen.isin(cm_pair)].index
        feature_train_pairs = feature_train_seen[cm_pair_index]
        y_train_pairs = list(y_train_seen[cm_pair_index])

        inter_dis(feature_train_pairs, labels=y_train_pairs,norm = True)
        inter_dis(feature_train_pairs, labels=y_train_pairs,norm = False)
        intra_var(feature_train_pairs, labels=y_train_pairs,norm = True)
        intra_var(feature_train_pairs, labels=y_train_pairs,norm = False)

    print("########### probs train  ########")
    var_and_mean_dis(prob_train_seen, labels=y_train_seen)
    y_test_seen = y_test_seen.tolist()
    print("########### feature test ########")
    var_and_mean_dis(feature_test_seen, labels=y_test_seen)
    print("########### probs test  ########")
    var_and_mean_dis(prob_test_seen, labels=y_test_seen)

    print("*********************************  pca *********************************")
    pca_labels = ["change_user_name","user_name","redeem_rewards","rewards_balance","ingredients_list","recipe","shopping_list","shopping_list_update","change_ai_name","what_is_your_name","play_music","change_speed"]
    pca_index = test_data[1][test_data[1].isin(pca_labels)].index
    print(feature_test[pca_index].shape)

    ids = le.transform(pca_labels)
    id2label = {}
    for i in range(len(pca_labels)):
        id2label[ids[i]]=pca_labels[i]
    print("id2label = ",id2label)
    pca_visualization(feature_test[pca_index],pd.Series(test_data[1])[pca_index],classes = pca_labels,save_path = "result/plot/pca2.png")
    plot_t_sne(feature_test[pca_index],pd.Series(test_data[1])[pca_index], sample_num = 30, id2label = id2label,save_path = "result/plot/pca0.png")

def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray) -> float:
    """
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
    lst = sorted(lst, key=lambda item: item[0])

    threshold = 0.
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0 # 起始阈值在最左边，所有数据都预测为unseen，（unseen视为正类positive）

    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        return (2 * p * r) / (p + r + 1e-10)

    f1 = compute_f1(tp, fp, fn)

    for m_dist, label in lst:   # 
        if label == "seen":  # fp -> tn
            fp -= 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            threshold = m_dist + 1e-10

    # print("estimated threshold:", threshold)
    return threshold

