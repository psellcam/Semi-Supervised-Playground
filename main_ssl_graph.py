# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:08:11 2022

@author: sella
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:30:42 2022

@author: sella
"""
import pandas as pd
import torch
import helpers
import matplotlib.pyplot as plt
from sklearn import datasets
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import random
import faiss
import scipy
from torch.nn import functional as F

if __name__ == '__main__':    
    
    X,Y = datasets.make_moons(n_samples=400, shuffle=False, noise=0.1 , random_state=1234)
    model = helpers.SmallNet()
    
    dataset = helpers.CustomMoonDataset(X, Y)
    all_indices = list(range(0,400))
    
    labelled_indices = random.sample(range(0, 200), 10) + random.sample(range(200, 400), 10)
    unlabelled_indices = [x for x in all_indices if x not in labelled_indices]
    
    #%%
       
    
    labelled_sampler = SubsetRandomSampler(labelled_indices)
    labelled_batch = BatchSampler(labelled_sampler, 5, drop_last=True)
    
    unlabelled_sampler = SubsetRandomSampler(unlabelled_indices)
    unlabelled_batch = BatchSampler(unlabelled_sampler, 15, drop_last=True)
            
    all_sampler = SubsetRandomSampler(all_indices)
    all_batch = BatchSampler(all_sampler, len(all_indices) , drop_last=False)
    
    #%%        
    labelled_loader = torch.utils.data.DataLoader(dataset,batch_sampler=labelled_batch,num_workers=3,pin_memory=True)
    unlabelled_loader = torch.utils.data.DataLoader(dataset,batch_sampler=unlabelled_batch,num_workers=3,pin_memory=True)
    all_loader = torch.utils.data.DataLoader(dataset,batch_sampler=all_batch,num_workers=3,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)       
    
    
    global_step = 0    
    #%%
    while(global_step < 300):
        new_labels = helpers.one_iter_true(X,Y, labelled_indices, unlabelled_indices)
        dataset.y = new_labels
        global_step = helpers.train_ssl_graph(labelled_loader, unlabelled_loader , model, optimizer, global_step)
        
        
    #%%   
        
    all_labels  , idx , predicts = helpers.train_eval(all_loader, model, optimizer) 
    all_labels = all_labels.numpy()
    idx = idx.numpy()
    predicts = predicts.numpy()
    
    #%%
    sorted_labels = np.zeros((400))          
    for i in range(400):
        sorted_labels[idx[i]] = predicts[i]
        
    
    #%%
    
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = Y))
    colors = {0 : 'red', 1 : 'blue'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',label=key,color=colors[key])
    plt.show()    
    
    #%%
    
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = sorted_labels))
    colors = {0 : 'red', 1 : 'blue'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.savefig("graph_pseudo.png", bbox_inches='tight',dpi=600)   
    
 