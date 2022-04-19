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
        return len(self.img_labels)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] , idx







class SmallNet(nn.Module):
    
    
    def __init__(self, num_classes=1):
        super(SmallNet, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.w1 = nn.Linear(2, 40)
        self.w2 = nn.Linear(40, 40)
        self.w3 = nn.Linear(40, 40)
        self.w4 = nn.Linear(40, num_classes)
        
    def forward(self, x, debug=False):
        
        output = self.w4(self.activation(self.w3(self.activation(self.w2(self.activation(self.w1(x.float())))))))
        return output
        
        
        
        
        
def train_sup(train_loader, model, optimizer, global_step):
    # switch to train mode
    model.train()
    labeled_iter = iter(train_loader)
    criterion = torch.nn.BCEWithLogitsLoss()
    p_bar = tqdm(range(100))
    losses = AverageMeter()
    
    for i in range(100):
        
        try:        
            inputs_x, targets_x , _ = labeled_iter.next()
        except:
            labeled_iter = iter(train_loader)
            inputs_x, targets_x , _ = labeled_iter.next()
        
        
        logits_x = model(inputs_x)      
        logits_x = torch.squeeze(logits_x)        
        loss = criterion(logits_x,targets_x.float())
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        global_step += 1
        p_bar.set_description("Loss: {loss:.4f}.".format(loss=losses.avg))
        p_bar.update()
                
                
                
    return global_step

def train_ssl(labelled_loader, unlabelled_loader, model, optimizer, global_step):
    # switch to train mode
    model.train()
    labeled_iter = iter(labelled_loader)
    unlabeled_iter = iter(unlabelled_loader)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    p_bar = tqdm(range(100))
    losses_l = AverageMeter()
    losses_u = AverageMeter()
    
    for i in range(100):
            
        try:        
            inputs_l, targets_l , _ = labeled_iter.next()
        except:
            labeled_iter = iter(labelled_loader)
            inputs_l, targets_l , _ = labeled_iter.next()
            
        try:        
            inputs_u, _ , _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabelled_loader)
            inputs_u, _ , _ = unlabeled_iter.next()
     
        
        logits_u = torch.squeeze(model(inputs_u))
        logits_l = torch.squeeze(model(inputs_l))
        
        
        targets_u = torch.round(torch.sigmoid(logits_u).detach())      
        
        
        loss_l = criterion(logits_l,targets_l.float())
        loss_u = criterion(logits_u,targets_u.float())
        
        loss = loss_l + 0.333*loss_u
        losses_l.update(loss_l.item())
        losses_u.update(loss_u.item())
        
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        global_step += 1
        p_bar.set_description("Loss_l: {loss_l:.4f}. Loss_u: {loss_u:.4f}".format(loss_l=losses_l.avg,loss_u=losses_u.avg))
        p_bar.update()
        
        
                
                
    return global_step



def train_ssl_graph(labelled_loader, unlabelled_loader, model, optimizer, global_step):
    # switch to train mode
    model.train()
    labeled_iter = iter(labelled_loader)
    unlabeled_iter = iter(unlabelled_loader)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    p_bar = tqdm(range(100))
    losses_l = AverageMeter()
    losses_u = AverageMeter()
    
    for i in range(100):
            
        try:        
            inputs_l, targets_l , _ = labeled_iter.next()
        except:
            labeled_iter = iter(labelled_loader)
            inputs_l, targets_l , _ = labeled_iter.next()
            
        try:        
            inputs_u, targets_u , _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabelled_loader)
            inputs_u, targets_u , _ = unlabeled_iter.next()
    
        
        logits_u = torch.squeeze(model(inputs_u))
        logits_l = torch.squeeze(model(inputs_l))        
        
        loss_l = criterion(logits_l,targets_l.float())
        loss_u = criterion(logits_u,targets_u.float())
        
        loss = loss_l + 0.333*loss_u
        losses_l.update(loss_l.item())
        losses_u.update(loss_u.item())
        
        loss.backward()
        optimizer.step()
        model.zero_grad()
        
        global_step += 1
        p_bar.set_description("Loss_l: {loss_l:.4f}. Loss_u: {loss_u:.4f}".format(loss_l=losses_l.avg,loss_u=losses_u.avg))
        p_bar.update()
        
        
                
                
    return global_step

        
def train_eval(test_loader, model, optimizer):
    # switch to train mode
    model.eval()
    labeled_iter = iter(test_loader)                
    inputs_x, targets_x , idx = labeled_iter.next()
    
    logits_x = model(inputs_x)      
    logits_x = torch.squeeze(logits_x)        
    logits_x = torch.sigmoid(logits_x)
    
    labels = torch.round(logits_x).int()
    result = torch.eq(torch.round(logits_x).int(),targets_x)
    
    
    print(result.sum().div(len(result)).item())
    
    return labels , idx , targets_x
    
    
    
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
        
        
        
def one_iter_true(X,Y, labelled_indices, unlabelled_indices,  k = 5, max_iter = 30):

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
    Z = np.zeros((N,2))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn        
    Y = np.zeros((N,2))
    Y[labelled_idx,labels[labelled_idx]] = 1      
    
    for i in range(2):
        f, _ = scipy.sparse.linalg.cg(A, Y[:,i], tol=1e-4, maxiter=max_iter)
        Z[:,i] = f
    Z[Z < 0] = 0 
        
    probs_iter = F.normalize(torch.tensor(Z),1).numpy()
    probs_iter[labelled_idx] = np.zeros(2) 
    probs_iter[labelled_idx,labels[labelled_idx]] = 1   
    
    p_labels = np.argmax(probs_iter,1)
    correct_idx = (p_labels[unlabelled_idx] == labels[unlabelled_idx])
    acc = correct_idx.mean()  
    print("Pseudo Label Accuracy {:.2f}".format(100*acc) + "%",end=" ") 
    return p_labels