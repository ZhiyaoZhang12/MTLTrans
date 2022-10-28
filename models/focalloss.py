# -*- coding: utf-8 -*-
from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):    
    def __init__(self, alpha, gamma, num_classes = 3, size_average=True):
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes              
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 
        self.gamma = gamma

    def forward(self, preds, labels): 
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))  
        factors = torch.pow((1-preds_softmax), self.gamma)
        focal_loss = -torch.mul(factors, preds_logsoft)  
        focal_loss = torch.mul(self.alpha, focal_loss.t()) 
        #shape  focal_loss:bs*1; factors:bs*1; pred_logsoft:bs*1
        if self.size_average:        
            focal_loss = focal_loss.mean()        
        else:            
            focal_loss = focal_loss.sum()        
        return focal_loss, factors