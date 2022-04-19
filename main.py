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

if __name__ == '__main__':    

    X,Y = datasets.make_moons(n_samples=400, shuffle=True, noise=0.1 , random_state=1234)
    model = helpers.SmallNet()
    
    dataset = helpers.CustomMoonDataset(X, Y)
    train_indices = list(range(0,320))
    test_indices = list(range(320,400))
    all_indices = list(range(0,400))
    
    
    #%%
    
    train_sampler = SubsetRandomSampler(train_indices)
    train_batch = BatchSampler(train_sampler, 20, drop_last=True)
    
    test_sampler = SubsetRandomSampler(test_indices)
    test_batch = BatchSampler(test_sampler, len(test_indices) , drop_last=False)
        
    all_sampler = SubsetRandomSampler(all_indices)
    all_batch = BatchSampler(all_sampler, len(all_indices) , drop_last=False)
    
    #%%        
    train_loader = torch.utils.data.DataLoader(dataset,batch_sampler=train_batch,num_workers=3,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset,batch_sampler=test_batch,num_workers=3,pin_memory=True)
    all_loader = torch.utils.data.DataLoader(dataset,batch_sampler=all_batch,num_workers=3,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)    
    
    
    global_step = 0
    
    while(global_step < 200):
        global_step = helpers.train_sup(train_loader, model, optimizer, global_step)
        
    all_labels  , idx , predicts = helpers.train_eval(all_loader, model, optimizer) 
    all_labels = all_labels.numpy()
    idx = idx.numpy()
    predicts = predicts.numpy()
    
    #%%
    sorted_labels = np.zeros((400))          
    for i in range(400):
        sorted_labels[idx[i]] = all_labels[i]
        
    
    #%%
    
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = Y))
    colors = {0 : 'red', 1 : 'blue'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.savefig("fully_labeled.png", bbox_inches='tight',dpi=600)
    #%%
    
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = sorted_labels))
    colors = {0 : 'red', 1 : 'blue'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.savefig("fully_supervised.png", bbox_inches='tight',dpi=600)   
    
 