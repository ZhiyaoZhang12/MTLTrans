# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import os


def plot_confusion_matrix(true,assess,title,normalize):
    
    #rcParams.update({'font.size': 14,  'font.family' : 'Times New Roman'}) 
    rcParams.update({'font.size': 14}) 
    

    #classes = ["Health","Incipient failure","Severe failure"]
    #classes = ["Class 0","Class 1","Class 2"]
    classes = ["0","1","2"]

    confusion = confusion_matrix(assess, true)
    
    if normalize:
        confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
    else:
        pass
    
    plt.imshow(confusion, cmap=plt.cm.Blues)
    
    indices = range(len(confusion))
    plt.xticks(indices, classes, rotation=0)  #rotation=0
    plt.yticks(indices, classes)
    plt.colorbar()
    
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            if normalize:
                plt.text(first_index, second_index, format(confusion[first_index][second_index], '.2f' ))  #.2f 
            else:
                plt.text(first_index, second_index, confusion[first_index][second_index])  

    plt.tight_layout()
    plt.title(title,loc='left')
    plt.ylabel('Ground truth')
    plt.xlabel('Assessment')

    
def SOH_confusion_matrix_fig(args,true,assess,soh_nll,test_dataset,flag):          
    fig=plt.figure(dpi=100,figsize=(16, 6))
    plt.subplots_adjust(wspace=4)
    plt.subplot(1,2,1)
    plot_confusion_matrix(true,assess,title='(a)',normalize=False)
    plt.subplot(1,2,2)
    plot_confusion_matrix(true,assess,title='(b)',normalize=True)
    
    fig_path = args.root_path + 'Fig/{}_{}/'.format(args.train_dataset_name,args.test_dataset_name)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path) 
    plt.savefig(fig_path + '{7}_SOH_{6}_{5}_a{0}gamma{1}_faultstart{2}_focalweight{3}_SOH confusion matrics_{4}.png'.format\
                                        (args.alpha_focal,args.gamma_focal,args.fault_start,args.focal_loss_weight,
                                                                                  soh_nll,flag,args.test_data,test_dataset,),bbox_inches='tight')
    #plt.show()