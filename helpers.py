# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:52:02 2022

@author: sella
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import faiss
from faiss import normalize_L2
import scipy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import itertools
import random
import numbers
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle

class StreamBatchSampler():

    def __init__(self, primary_indices, batch_size):
        self.primary_indices = np.asarray(primary_indices)
        self.primary_batch_size = batch_size

    def __iter__(self):
        primary_iter = iterate_eternally(self.primary_indices)
        return (primary_batch  for (primary_batch)
            in  grouper(primary_iter, self.primary_batch_size)
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
    
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())    
    
def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)         




class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        

class CustomMoonDataset(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] 



class SmallNet(nn.Module):       
    def __init__(self, num_classes=2):
        super(SmallNet, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.w1 = nn.Linear(2, 50)
        self.w2 = nn.Linear(50, num_classes)
        
    def forward(self, x, debug=False):        
        output = self.w1(x.float())
        return self.w2(self.activation(output)) 
        

        
def train_sup(train_loader, model, optimizer, global_step):
    # switch to train mode
    model.train()
    labeled_iter = iter(train_loader)
    criterion = torch.nn.CrossEntropyLoss()
    p_bar = tqdm(range(3000))
    losses = AverageMeter()
    
    for i in range(3000):         
        inputs_x, targets_x  = labeled_iter.next()      
        
        logits_x = model(inputs_x)    
        loss = criterion(logits_x,targets_x)
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        global_step += 1
        p_bar.set_description("Loss: {loss:.4f}.".format(loss=losses.avg))
        p_bar.update()               
    return global_step


def train_eval(dataset, model, optimizer,test_indices):
    # switch to train mode
    model.eval()    
    logits_x  = model(torch.from_numpy(dataset.x))
    target_x = torch.from_numpy(dataset.y)    
    labels = torch.argmax(logits_x, dim=1).int()
    result = torch.eq(labels,target_x)        
    print(result[test_indices].sum().div(len(result[test_indices])).item())
                
    return labels.numpy() 



def train_ssl(labelled_loader, unlabelled_loader, model, optimizer, global_step,batchsize):
    # switch to train mode
    model.train()
    labeled_iter = iter(labelled_loader)
    unlabeled_iter = iter(unlabelled_loader)    
    criterion = torch.nn.CrossEntropyLoss()
    p_bar = tqdm(range(3000))
    losses_l = AverageMeter()
    losses_u = AverageMeter()
    
    for i in range(3000):
            
        inputs_l, targets_l  = labeled_iter.next()       
        inputs_u, _  = unlabeled_iter.next()        
        
        logits_l = model(inputs_l)  
        logits_u = model(inputs_u)
           
        
        pseudo_label = torch.softmax(logits_u.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(0.8).float()
        
        loss_l = F.cross_entropy(logits_l, targets_l, reduction='mean')
        loss_u = (F.cross_entropy(logits_u, targets_u, reduction='none') * mask).mean()         
                       
        loss = loss_l + loss_u
        losses_l.update(loss_l.item())
        losses_u.update(loss_u.item())
        
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        global_step += 1
        p_bar.set_description("Loss_l: {loss_l:.4f}. Loss_u: {loss_u:.4f}".format(loss_l=losses_l.avg,loss_u=losses_u.avg))
        p_bar.update()       
                
                
    return global_step

def train_ssl_graph(labelled_loader, unlabelled_loader, model, optimizer, global_step,batchsize):
    # switch to train mode
    model.train()
    labeled_iter = iter(labelled_loader)
    unlabeled_iter = iter(unlabelled_loader)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    p_bar = tqdm(range(3000))
    losses_l = AverageMeter()
    losses_u = AverageMeter()
    
    for i in range(3000):               
        inputs_l, targets_l  = labeled_iter.next()   
        inputs_u, targets_u  = unlabeled_iter.next()
    
        
        logits_u  = model(inputs_u)
        logits_l  = model(inputs_l)        
        
        loss_l = criterion(logits_l,targets_l)
        loss_u = criterion(logits_u,targets_u)
        
        loss = (loss_l + loss_u)/batchsize
        losses_l.update(loss_l.item())
        losses_u.update(loss_u.item())
        
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        global_step += 1
        p_bar.set_description("Loss_l: {loss_l:.4f}. Loss_u: {loss_u:.4f}".format(loss_l=losses_l.avg,loss_u=losses_u.avg))
        p_bar.update()
        
        
                
                
    return global_step

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def adjust_learning_rate(optimizer, global_step):
    
    lr = 0.005
    lr *= cosine_rampdown(global_step, 1100)
    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def sparse_indices(Y,train_indices,n_samples,n_each):
    train_labels = Y[train_indices]
    class_0 = [i for i in range(train_labels.shape[0]) if train_labels[i] == 0]
    random.shuffle(class_0)
    class_1 = [i for i in range(train_labels.shape[0]) if train_labels[i] == 1]
    random.shuffle(class_1)
    class_2 = [i for i in range(train_labels.shape[0]) if train_labels[i] == 2]
    random.shuffle(class_2)
    
    train_indices_ssl = np.sort(class_0[:n_each] + class_1[:n_each] + class_2[:n_each])   
    Y_ssl = -1*np.ones(n_samples)
    Y_ssl[train_indices_ssl] = Y[train_indices_ssl]
    
    return Y_ssl , train_indices_ssl 

        
        
def one_iter_true(X,Y, labelled_indices, unlabelled_indices,k = 10, max_iter = 300 , num_classes =3):

    alpha = 0.99
    X = np.ascontiguousarray(X)
    X = np.float32(X)
    labels = np.asarray(Y)
    labelled_idx = np.asarray(labelled_indices)        
    unlabelled_idx = np.asarray(unlabelled_indices)
    
    d = X.shape[1]   
    index = faiss.IndexFlatL2(d)    
    index.add(X)    
    N = X.shape[0]
    D, I = index.search(X, k + 1)
    D = np.exp(-D)
    
    
    # Create the graph
    D = D[:,1:] 
    I = I[:,1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))       
    W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)       
   
    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D    
    
    # Initiliaze the y vector for each class          
    Z = np.zeros((N,num_classes))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
    Y = np.zeros((N,num_classes))
    Y[labelled_idx,labels[labelled_idx]] = 1      
    
    for i in range(num_classes):
        f, _ = scipy.sparse.linalg.cg(A, Y[:,i], tol=1e-4, maxiter=max_iter)
        Z[:,i] = f
    Z[Z < 0] = 0 
        
    probs_iter = F.normalize(torch.tensor(Z),1).numpy()
    probs_iter[labelled_idx] = np.zeros(num_classes) 
    probs_iter[labelled_idx,labels[labelled_idx]] = 1   
    
    p_labels = np.argmax(probs_iter,1)
    correct_idx = (p_labels[unlabelled_idx] == labels[unlabelled_idx])
    acc = correct_idx.mean()  
    print("Pseudo Label Accuracy {:.2f}".format(100*acc) + "%",end=" ") 
    return p_labels