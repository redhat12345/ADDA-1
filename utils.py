import matplotlib.pyplot as plt  
import numpy as np 
from sklearn.manifold import TSNE
import os
import tensorflow as tf 

# plot accuracy
def plot_acc(acc_list,threshold=0.7,name="target domain accuracy"):
    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.plot(range(len(acc_list)),acc_list)
    for idx,acc in enumerate(acc_list):
        if acc>0.7:
            plt.text(idx,acc,str(acc))

def plot_tsne_orign(source_img,source_label,target_img,target_label,samples=None,name="Samples_before_adaptation"):
    source_img = source_img.reshape((source_img.shape[0],-1))
    target_img = target_img.reshape((target_img.shape[0],-1))
    plot_tsne(source_img,source_label,target_img,target_label,samples,name)

def plot_tsne(source_feat,source_label,target_feat,target_label,samples=None,name="Samples_after_adaptation"):
    if samples!=None:
        each_class_num = int(samples / 10)
        #source_feat = source_feat[:samples,:]
        #source_label = source_label[:samples]
        #target_feat = target_feat[:samples,:]
        #target_label = target_label[:samples]
        source_feat,source_label = select_samples(source_feat,source_label,each_class_num)
        target_feat,target_label = select_samples(target_feat,target_label,each_class_num)
        print(source_feat.shape)
    
    feat = np.concatenate([source_feat,target_feat],axis=0)
    tsne = TSNE(n_components=2,random_state=0)
    reduce_feat = tsne.fit_transform(feat)
    
    reduce_source_feat = reduce_feat[:source_feat.shape[0],:]
    reduce_target_feat = reduce_feat[source_feat.shape[0]:,:]
    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    plot_embedding(ax,reduce_source_feat,source_label,0)
    plot_embedding(ax,reduce_target_feat,target_label,1)
    ax.set_xticks([]),ax.set_yticks([])
    ax.legend()
    plt.savefig("./result/"+name+".jpg")

def select_samples(x,y,each_sample):
    ind = []
    count = [0]*10
    for id,it in enumerate(y):
        if count[it] < each_sample:
            ind.append(id)
            count[it] += 1
        if sum(count) == each_sample * 10:
            break
    #print(ind)
    return x[ind],y[ind]


def plot_embedding(ax,X,y,d):
    x_min,x_max = np.min(X,0),np.max(X,0)
    X = (X-x_min)/(x_max-x_min)

    # plot color numbers
    if d==0:
        label="source"
    else:
        label="target"
    ax.scatter(X[:,0],X[:,1],marker='.',color=plt.cm.bwr(d / 1.),label=label)
    for i in range(X.shape[0]):
        ax.text(X[i,0],X[i,1],str(y[i]),color=plt.cm.bwr(d / 1.), fontdict={"weight":"bold","size":9})
    
# clear old path,and build new one
def fresh_dir(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
    tf.gfile.MakeDirs(path)    
