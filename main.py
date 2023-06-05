import argparse
import json
import os
import random
import datetime

import numpy as np
import umap
from numpy import ndarray
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import load_yaml, merge_arg
from evaluation import bcubed, pairwise
from utils import read
import matplotlib.pyplot as plt
import time


def parse_args():
    '''添加运行参数'''
    parser=argparse.ArgumentParser(description="Add configs")
    parser.add_argument("--config",type=str,default="config/config.yaml",help="Path to config file")
    parser.add_argument("--mode",type=str,default="run",help="Mode of the experiment")
    parser.add_argument("--method",type=str,default="K-means",help="Method of the experiment")
    parser.add_argument("--is_normalize",type=bool,default=True,help="Whether normalize the features")
    parser.add_argument("--normalize_method",type=str,default="min-max",help="Method of normalization")
    parser.add_argument("--decomposition_method",type=str,default="PCA",help="Method of decomposition")
    parser.add_argument("--select_num",type=int,default=10,help="Select cluster number")
    args=parser.parse_args()
    return args

def min_max_normalize(feat:ndarray):
    '''min-max归一化'''
    feat_min=np.min(feat)
    feat_max=np.max(feat)
    feat=(feat-feat_min)/(feat_max-feat_min)
    return feat

def z_score_normalize(feat:ndarray):
    '''z-score归一化'''
    feat_mean=np.mean(feat)
    feat_std=np.std(feat)
    feat=(feat-feat_mean)/feat_std
    return feat

def normalize(feat:ndarray,normalize_method:str):
    '''归一化方法统一接口'''
    if normalize_method=="min-max":
        feat=min_max_normalize(feat)
    elif normalize_method=="z-score":
        feat=z_score_normalize(feat)
    else:
        raise ValueError("Normalize method error: %r is not available" % normalize_method)
    return feat

class Cluster(object):
    '''统一聚类算法接口'''
    def __init__(self,method:str,cluster_num:int):
        if method=="K-means":
            self.cluster=KMeans(n_clusters=cluster_num)
        elif method=="HAC":
            self.cluster=AgglomerativeClustering(n_clusters=cluster_num)
        elif method=="DBSCAN":
            self.cluster=DBSCAN(eps=10,min_samples=2)
        elif method=="Spectral":
            self.cluster=SpectralClustering(n_clusters=cluster_num,affinity='rbf',n_jobs=10)
        else:
            raise ValueError("Method error: %r is not available" % method)


    def fit_predict(self,feat:ndarray):
        '''对数据进行聚类并进行预测'''
        return self.cluster.fit_predict(feat)


class Decomposition(object):
    '''统一降维算法接口'''
    def __init__(self,decomposition_method:str):
        if decomposition_method=="PCA":
            self.decomposition=PCA(n_components=2)
        elif decomposition_method=="UMAP":
            self.decomposition=umap.UMAP(n_components=2)
        elif decomposition_method=="t-SNE":
            self.decomposition=TSNE(n_components=2)
        else:
            raise ValueError("decomposition method error: %r is not available" % decomposition_method)

    def fit_transform(self,feat:ndarray):
        '''对数据进行降维'''
        return self.decomposition.fit_transform(feat)

def cal_score(label:ndarray,predict_label:ndarray,parameter_path:str):
    '''对预测的标签进行评分'''
    # 获得预测结果的Bcubed F-score以及precision和recall
    pre, recall, fb = bcubed(gt_labels=label, pred_labels=predict_label)
    bcubed_dict = {'F-score':fb,'precision':pre,'recall':recall}
    print("Bcubed F-score:{}, precision:{}, recall:{}".format(fb, pre, recall))
    # 获得预测结果的Pairwise F-score以及precision和recall
    pre, recall, fp = pairwise(gt_labels=label, pred_labels=predict_label)
    pairwise_dict = {'F-score': fp, 'precision': pre, 'recall': recall}
    print("Pairwise F-score:{}, precision:{}, recall:{}".format(fp, pre, recall))
    result_dict={'Bcubed':bcubed_dict,'Pairwise':pairwise_dict}
    with open(parameter_path,"a",encoding='utf-8') as f:
        f.write(json.dumps(result_dict,ensure_ascii=False,indent=4))
        f.write('\n')


def train_and_test(feat:ndarray, label:ndarray, method:str, is_normalize:bool, normalize_method:str, cluster_num:int,parameter_path:str):
    '''训练并进行测试'''
    if is_normalize:
        feat=normalize(feat,normalize_method)
    cluster=Cluster(method,cluster_num)
    predict_label=cluster.fit_predict(feat)
    cal_score(label,predict_label,parameter_path)

def visualization(feat:ndarray,label:ndarray,method:str,is_normalize:bool,normalize_method:str,cluster_num:int,decomposition_method:str,save_path:str):
    '''对数据进行可视化处理'''
    if is_normalize:
        feat=normalize(feat,normalize_method)
    cluster=Cluster(method,cluster_num)
    predict_label=cluster.fit_predict(feat)
    decomposition=Decomposition(decomposition_method)
    decomposition_feat=decomposition.fit_transform(feat)
    colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    x0_max=-99999
    x0_min=99999
    x1_max=-99999
    x1_min=99999
    plt.rcParams['savefig.dpi']=300
    for i in range(cluster_num):
        index = np.nonzero(predict_label == i)[0]
        x0 = decomposition_feat[index, 0]
        x1 = decomposition_feat[index, 1]
        x0_max=max(np.max(x0),x0_max)
        x0_min = min(np.min(x0), x0_min)
        x1_max = max(np.max(x1), x1_max)
        x1_min = min(np.min(x1), x1_min)
        y_i = label[index]
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(int(y_i[j])), fontdict={'color':colors[i],'ha':'center','va':'center','size':9})
    plt.scatter(x0_max, x1_max, alpha=0)
    plt.scatter(x0_min,x1_min,alpha=0)
    plt.savefig(save_path,format='svg')
    #plt.show()


def select(feat:ndarray,label:ndarray,select_num:int,cluster_num:int):
    '''选择10个种类数据'''
    selected_feat=[]
    selected_label=[]
    selected_nums=[]
    for i in range(select_num):
        selected_num=random.randint(0,cluster_num-1)
        while selected_num in selected_nums :
            selected_num = random.randint(0, cluster_num - 1)
        selected_nums.append(selected_num)
    for i in range(len(feat)):
        if label[i] in selected_nums:
            selected_feat.append(feat[i])
            selected_label.append(label[i])
    selected_feat=np.array(selected_feat)
    selected_label=np.array(selected_label)
    return selected_feat,selected_label


def main():
    begin=datetime.datetime.now()
    args = parse_args()
    config = load_yaml(args.config)
    args = merge_arg(args, config)
    mode = args.mode
    method = args.method
    select_num=args.select_num
    decomposition_method=args.decomposition_method
    if mode=="run":
        cluster_num=3991
    elif mode=="debug":
        cluster_num=1000
    else:
        raise ValueError("Mode error: %r is not available" % mode)
    is_normalize = args.is_normalize
    normalize_method=args.normalize_method
    feat_path = 'feat.bin'
    feat_subset_path = 'feat_subset.bin'
    label_path = 'label.meta'
    label_subset_path = 'label_subset.meta'
    feat, label = read(feat_path, label_path)
    feat_subset, label_subset = read(feat_subset_path, label_subset_path)
    timestamp=time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    if is_normalize:
        save_path=os.path.join("result",method+' '+normalize_method+' '+decomposition_method+' '+str(timestamp))
    else:
        save_path = os.path.join("result",
                                 method + ' ' + 'none' + ' ' + decomposition_method + ' ' + str(timestamp))
    os.makedirs(save_path,exist_ok=True)
    img_save_path=os.path.join(save_path,"result.svg")
    parameter_path=os.path.join(save_path,"parameter.json")
    with open(parameter_path,"a",encoding='utf-8') as f:
        f.write(json.dumps(vars(args),ensure_ascii=False,indent=4))
        f.write('\n')
    if mode == "run":
        train_and_test(feat, label, method, is_normalize, normalize_method, cluster_num,parameter_path)
        if method=="K-means" or method=="HAC" or method=="Spectral":
            selected_feat,selected_label=select(feat,label,select_num,cluster_num)
            visualization(selected_feat,selected_label,method,is_normalize,normalize_method,select_num,decomposition_method,img_save_path)
    elif mode == "debug":
        train_and_test(feat_subset, label_subset, method, is_normalize, normalize_method, cluster_num,parameter_path)
        if method=="K-means" or method=="HAC" or method=="Spectral":
            selected_feat_subset,selected_label_subset=select(feat_subset,label_subset,select_num,cluster_num)
            visualization(selected_feat_subset,selected_label_subset,method,is_normalize,normalize_method,select_num,decomposition_method,img_save_path)
    else:
        raise ValueError("Mode error: %r is not available" % mode)
    end=datetime.datetime.now()
    process_time=(end-begin).seconds
    time_dict={'time':process_time}
    with open(parameter_path,"a",encoding='utf-8') as f:
        f.write(json.dumps(time_dict,ensure_ascii=False,indent=4))

if __name__ == '__main__':
    main()