# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:30:42 2022

@author: sella
"""
import pandas as pd
import torch
import helpers
import random
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

if __name__ == '__main__':    

    ### Let us make the synthetic two moons dataset
    n_samples = 500    
    X,Y = make_moons(n_samples=n_samples, shuffle=True, noise=0.10 , random_state=5)    

    indices = np.arange(Y.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    dataset = helpers.CustomMoonDataset(X, Y)
    train_indices = list(range(0,int(0.8*n_samples)))
    test_indices = list(range(int(0.8*n_samples),n_samples))   
    
    
    #%% Plot to show the nature of the data    
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = Y))
    colors = { 0 : 'red', 1 : 'blue' , 2 : 'green' , 3 : 'yellow'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.show()
    
    
    
    #%% Fully Supervised Training    
    batchsize = 20
    train_sampler = helpers.StreamBatchSampler(train_indices, batchsize)        
    model = helpers.SmallNet()    
    train_loader = torch.utils.data.DataLoader(dataset,batch_sampler=train_sampler,num_workers=3,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001)    
    
    
    global_step = 0    
    global_step = helpers.train_sup(train_loader, model, optimizer, global_step)        
    predicts = helpers.train_eval(dataset, model, optimizer,test_indices)
        
    

    #### Result     
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = predicts))
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.show() 
    
    
    #%% Sparse Supervised Training
    Y_sparse , train_indices_sparse  = helpers.sparse_indices(Y,train_indices,n_samples,4)

    
    
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = Y_sparse))
    colors = {-1: 'grey' , 0 : 'red', 1 : 'blue', 2 : 'green'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.show() 
    
    
    
    #%% Sparse Supervised Training    
    train_sampler_sparse = helpers.StreamBatchSampler(train_indices_sparse, 5)        
    train_loader = torch.utils.data.DataLoader(dataset,batch_sampler=train_sampler_sparse,num_workers=3,pin_memory=True)
    model = helpers.SmallNet()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001)    
    
    
    global_step = 0    
    global_step = helpers.train_sup(train_loader, model, optimizer, global_step)        
    predicts_sparse = helpers.train_eval(dataset, model, optimizer,test_indices)
    
    
    ##### Result     
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = predicts_sparse))
    colors = {-1: 'grey' , 0 : 'red', 1 : 'blue', 2 : 'green'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.show() 
    
    
    #%% Can SSL do better?
    labelled_indices = train_indices_sparse
    unlabelled_indices = np.asarray([i for i in range(n_samples) if i not in train_indices_sparse and i not in test_indices])
    
    labelled_sampler = helpers.StreamBatchSampler(labelled_indices, 5) 
    unlabelled_sampler = helpers.StreamBatchSampler(unlabelled_indices, 15)  
    labelled_loader = torch.utils.data.DataLoader(dataset,batch_sampler=labelled_sampler,num_workers=3,pin_memory=True)
    unlabelled_loader = torch.utils.data.DataLoader(dataset,batch_sampler=unlabelled_sampler,num_workers=3,pin_memory=True)
    
    
    #%% Semi-Supervised Training
    
    model = helpers.SmallNet()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001)    
    global_step = 0    
    global_step = helpers.train_ssl(labelled_loader, unlabelled_loader, model, optimizer, global_step,batchsize)
    predicts_sparse = helpers.train_eval(dataset, model, optimizer, test_indices)
    
    
    
    ##### Result     
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = predicts_sparse))
    colors = {0 : 'red', 1 : 'blue' , 2 : 'green'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.show() 
    
    
    #%% Semi-Supervised Training Graph Style 
               
    model = helpers.SmallNet()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001)    
    global_step = 0         
    new_labels = helpers.one_iter_true(X,Y, labelled_indices, unlabelled_indices,k=5)
    dataset.y = new_labels             
    global_step = helpers.train_ssl_graph(labelled_loader, unlabelled_loader, model, optimizer, global_step,batchsize)
    predicts_sparse_graph = helpers.train_eval(dataset, model, optimizer, test_indices)
    
    
    #### Result     
    df = pd.DataFrame(dict(x = X[:,0] , y = X[:,1] , label = predicts_sparse_graph))
    colors = {0 : 'red', 1 : 'blue' , 2 : 'green'}
    fig,ax = plt.subplots()
    grouped = df.groupby('label')
    for key,group in grouped:
        group.plot(ax=ax,kind='scatter',x='x',y='y',color=colors[key])
    plt.axis('off')
    plt.show() 
    