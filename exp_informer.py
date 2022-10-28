# -*- coding: utf-8 -*-
from data.data_loader import DataReaderTrajactory,DataPrepare,TrainValiTest
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack  
from sklearn.preprocessing import LabelBinarizer
from utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import warnings
import torch.nn.functional as F
from models.focalloss import FocalLoss 
from models.rul_process import RUL_results_save, Draw_RUL_decreasing_fig,Draw_RUL_unit_fig
from models.soh_process import SOH_confusion_matrix_fig
from models.score_fun import score_loss, score_focal_loss
from utils.metrics import metric
from sklearn.metrics import classification_report,precision_recall_fscore_support

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        
        args = self.args
        
        dataset_used = list(set(args.train_dataset_name + args.test_dataset_name)) 
        dataset_used.sort()
        
        for dataset in dataset_used: 
            data_prepare = DataPrepare(root_path=args.root_path,
                       HI_labeling_style = args.HI_labeling_style,
                       dataset_name=dataset, 
                       test_data=args.test_data,
                       sensor_features=args.sensor_features,
                       OP_features = args.OP_features,
                       normal_style=args.normal_style,
                       pred_len=args.pred_len,seq_len=args.seq_len,
                       fault_start= args.fault_start)
            data_prepare.process()
         
       

        train_vali_test  = TrainValiTest(
                            root_path=args.root_path,
                            train_dataset_name=args.train_dataset_name,
                            test_dataset_name=args.test_dataset_name,
                            seq_len=args.seq_len,
                            pred_len=args.pred_len,
                            label_len=args.label_len,
                            sensor_features=args.sensor_features,
                            is_padding=args.is_padding,
                            is_descrsing=args.is_descrsing,
                            data_augmentation=args.data_augmentation,
                            HI_labeling_style=args.HI_labeling_style,
                            normal_style = args.normal_style,
                            synthetic_data_path = args.synthetic_data_path)
        train_vali_test.spilt_data()
        
   
    def _build_model(self):

        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args,
                self.args.OP_features,
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.is_perception,
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
                

            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
        

    def _get_data(self, flag, dataset_name):
        args = self.args
      
        if flag in ['test_window','test_whole']:  
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size 
        elif flag=='pred':   
            shuffle_flag = False; drop_last = False; batch_size = 20
        else:  #train,val
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
       
        Data =  DataReaderTrajactory 
        data_set = Data(
            root_path=args.root_path,
            train_dataset_name=args.train_dataset_name,
            test_dataset_name=args.test_dataset_name,
            dataset_name=dataset_name,
            sensor_features=args.sensor_features,
            is_padding=args.is_padding,
            is_descrsing=args.is_descrsing,
            data_augmentation=args.data_augmentation,
            HI_labeling_style=args.HI_labeling_style,
            normal_style = args.normal_style,
            synthetic_data_path = args.synthetic_data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            target=args.target,
            inverse=args.inverse,
            cols=args.cols
        )
        
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    
    def _select_criterion_hi(self):
        if self.args.loss_hi == 'mse':
            if self.args.focal_loss_weight == True:
                criterion_hi =  nn.MSELoss(reduction='none')   #default reduction='mean'  
            else:
                criterion_hi =  nn.MSELoss(reduction='mean')
        elif self.args.loss_hi == 'mae':
            criterion_hi =  nn.L1Loss()
        return criterion_hi
    
    def _select_criterion_rul(self):
        if self.args.loss_rul == 'mse':
            if self.args.focal_loss_weight == True:
                criterion_rul =  nn.MSELoss(reduction='none')
            else:
                criterion_rul =  nn.MSELoss(reduction='mean')
        elif self.args.loss_rul == 'mae':
            criterion_rul =  nn.L1Loss()
        elif self.args.loss_rul == 'score':            
            if self.args.focal_loss_weight == True:
                 criterion_rul = score_loss
            else:
                 criterion_rul = score_focal_loss               
        return criterion_rul
    
    def _select_criterion_soh(self):
        if self.args.loss_soh == 'cross_entropy':
            criterion_soh = F.cross_entropy
        if self.args.loss_soh == 'NLL': 
            criterion_soh = nn.NLLLoss()
        if self.args.loss_soh == 'focal_loss':
            focal_loss = FocalLoss(alpha=self.args.alpha_focal,gamma=self.args.gamma_focal,num_classes=3,size_average=True)
            criterion_soh = focal_loss
        return criterion_soh
    
    # sortmax 结果转 onehot
    def _props_to_onehot(self,props):
        if isinstance(props, list):
            props = np.array(props)
        a = np.argmax(props, axis=1)
        b = np.zeros((len(a), props.shape[1]))
        b[np.arange(len(a)), a] = 1
        return torch.tensor(b)
    

    def vali_MTL(self, vali_data, vali_loader, criterion_hi, criterion_rul, criterion_soh):
        self.model.eval()
        with torch.no_grad():  #for saving time and capacity
            #GPU tensor
            device = torch.device('cuda:0')
            total_loss_hi,total_loss_rul,total_loss_soh,total_factors = torch.tensor([]).to(device),torch.tensor([]).to(device),torch.tensor([]).to(device),torch.tensor([]).to(device)

            for i, (batch_x,batch_y,) in enumerate(vali_loader):

                if self.args.output_attention:
                    Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, attn_weights, indexs = \
                                    self._process_one_batch_MTL(vali_data, batch_x, batch_y)
                else:
                    Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, indexs = \
                                    self._process_one_batch_MTL(vali_data, batch_x, batch_y)

                if self.args.is_perception == False:
                    loss_hi = criterion_hi(Y_HI_out, Y_HI)
                    loss_rul = criterion_rul(Y_RUL_out, Y_RUL)
                    if self.args.loss_soh == 'focal_loss':
                        loss_soh,factors = criterion_soh(Y_SOH_out, Y_SOH)  
                    else:
                        loss_soh = criterion_soh(Y_SOH_out, Y_SOH)       

                elif self.args.is_perception == True:
                    loss_hi = criterion_hi(Y_HI_out[:,-self.args.seq_len:,:],Y_HI[:,-self.args.seq_len:,:])
                    loss_rul = criterion_hi(Y_RUL_out[:,-self.args.seq_len:,:],Y_RUL[:,-self.args.seq_len:,:]) 
                    if self.args.loss_soh == 'focal_loss':
                        loss_soh,factors = criterion_soh(Y_SOH_out[:,-self.args.seq_len:,:],Y_SOH[:,-self.args.seq_len:,:])
                    else:
                        loss_soh = criterion_soh(Y_SOH_out[:,-self.args.seq_len:,:],Y_SOH[:,-self.args.seq_len:,:])

                if self.args.focal_loss_weight==True:
                    loss_hi = loss_hi.mean(axis=1) #loss_hi bs*pred_len*1 --> bs*1

                total_loss_hi=torch.cat([total_loss_hi,loss_hi.reshape([-1])])  
                total_loss_soh=torch.cat([total_loss_soh,loss_soh.reshape([-1])])
                total_loss_rul=torch.cat([total_loss_rul,loss_rul.reshape([-1])])

                if self.args.loss_soh == 'focal_loss':
                    total_factors=torch.cat([total_factors,factors])             

            if self.args.loss_soh == 'focal_loss':
                total_factors = torch.tensor(total_factors)

            total_loss_hi = torch.mean(total_loss_hi)
            total_loss_soh = torch.mean(total_loss_soh) 
            total_loss_rul = torch.mean(total_loss_rul)    

            total_loss = self.args.loss_weight[0]*total_loss_hi + self.args.loss_weight[1]*total_loss_rul + self.args.loss_weight[2]*total_loss_soh      
            self.model.train()
            return total_loss, total_loss_rul, total_loss_soh, total_loss_hi


   
    def train_MTL(self, setting):
        train_data, train_loader = self._get_data(flag='train',dataset_name=self.args.train_dataset_name)
        vali_data, vali_loader = self._get_data(flag='val',dataset_name=self.args.train_dataset_name)

              
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        

        model_optim = self._select_optimizer()
        criterion_hi =  self._select_criterion_hi()
        criterion_rul =  self._select_criterion_rul()
        criterion_soh =  self._select_criterion_soh()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1 
                model_optim.zero_grad()  

                if self.args.output_attention:
                    Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, attn_weight, indexs = \
                                        self._process_one_batch_MTL(train_data, batch_x, batch_y)                   
                else:
                    Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, indexs = \
                                self._process_one_batch_MTL(train_data, batch_x, batch_y)

                if self.args.is_perception == False:
                    loss_hi = criterion_hi(Y_HI_out, Y_HI)
                    loss_rul = criterion_rul(Y_RUL_out, Y_RUL)
                    if self.args.loss_soh == 'focal_loss':
                        loss_soh,factors =  criterion_soh(Y_SOH_out,Y_SOH)   #(bs*3)  & (bs,)  
                    else:
                        loss_soh = criterion_soh(Y_SOH_out,Y_SOH)   #(bs*3)  & (bs,)


                elif self.args.is_perception == True:
                    loss_hi = criterion_hi(Y_HI_out[:,-self.args.seq_len:,:], Y_HI[:,-self.args.seq_len:,:])
                    loss_rul = criterion_rul(Y_RUL_out[:,-self.args.seq_len:,:], Y_RUL[:,-self.args.seq_len:,:])
                    if self.args.loss_soh == 'focal_loss':
                        loss_soh,factors = criterion_hi(Y_SOH_out[:,-self.args.seq_len:,:],Y_SOH[:,-self.args.seq_len:,:])
                    else:
                        loss_soh = criterion_hi(Y_SOH_out[:,-self.args.seq_len:,:],Y_SOH[:,-self.args.seq_len:,:])


                if self.args.focal_loss_weight == True:
                    loss_rul = torch.mul(factors, loss_rul).mean()
                    loss_hi = torch.mul(factors, loss_hi.mean(axis=1)).mean()
                else:
                    loss_rul = loss_rul.mean()
                    loss_hi = loss_hi.mean()


                loss = self.args.loss_weight[0]*loss_hi + self.args.loss_weight[1]*loss_rul + self.args.loss_weight[2]*loss_soh

                train_loss.append(loss.item())

                if (i+1) % 50==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print("\titers: {0}, epoch: {1} | loss_hi: {2:.7f}".format(i + 1, epoch + 1, loss_hi.item()))
                    print("\titers: {0}, epoch: {1} | loss_rul: {2:.7f}".format(i + 1, epoch + 1, loss_rul.item()))
                    print("\titers: {0}, epoch: {1} | loss_soh: {2:.7f}".format(i + 1, epoch + 1, loss_soh.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))

            train_loss = np.average(train_loss)

            vali_loss, vali_loss_rul, vali_loss_soh, vali_loss_hi = self.vali_MTL(vali_data, vali_loader, criterion_hi, criterion_rul, criterion_soh)

            print("Epoch: {0}, Steps: {1} | Train loss: {2:.7f} Vali loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            print("Epoch: {0}, Steps: {1} |  Vali loss HI: {2:.7f} ".format(
                epoch + 1, train_steps,  vali_loss_hi))
            print("Epoch: {0}, Steps: {1} |  Vali loss RUL: {2:.7f}".format(
                epoch + 1, train_steps,  vali_loss_rul))
            print("Epoch: {0}, Steps: {1} |  Vali loss SOH: {2:.7f}".format(
                epoch + 1, train_steps,  vali_loss_soh))

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path + '/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
        
        

    def test_MTL(self, setting, flag, test_dataset):
        test_data, test_loader = self._get_data(flag=flag,dataset_name=test_dataset)
      
        self.model.eval()
        with torch.no_grad():  
            Y_HI_outs, Y_RUL_outs, Y_SOH_outs = torch.tensor([]),torch.tensor([]),torch.tensor([])
            Y_HIs, Y_RULs, Y_SOHs = torch.tensor([]),torch.tensor([]),torch.tensor([])
            indexss = torch.tensor([])

            for i, (batch_x,batch_y) in enumerate(test_loader):
                if self.args.output_attention==True:
                    Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, attn_weights, indexs = self._process_one_batch_MTL(test_data, batch_x, batch_y)
                   
                elif self.args.output_attention==False:

                    Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, indexs = self._process_one_batch_MTL(test_data, batch_x, batch_y)
                 

                Y_HI_outs = torch.cat([Y_HI_outs,Y_HI_out.detach().cpu()])   
                Y_SOH_outs = torch.cat([Y_SOH_outs,Y_SOH_out.detach().cpu()])
                Y_HIs = torch.cat([Y_HIs,Y_HI.detach().cpu()])
                Y_SOHs = torch.cat([Y_SOHs,Y_SOH.detach().cpu()])                
                Y_RUL_outs = torch.cat([Y_RUL_outs,Y_RUL_out.detach().cpu()])
                Y_RULs = torch.cat([ Y_RULs,Y_RUL.detach().cpu()])      
                indexss = torch.cat([indexss,indexs.detach().cpu()])


            Y_HI_outs = np.array(Y_HI_outs)           
            Y_SOH_outs = np.array(Y_SOH_outs)
            Y_HIs = np.array(Y_HIs)
            Y_SOHs = np.array(Y_SOHs)
            Y_RULs = np.array(Y_RULs)
            Y_RUL_outs = np.array(Y_RUL_outs)
            indexss = np.array(indexss)

            #shape   RUL num_batch*batch_size*1    SOH num_batch*batch_size*3   HI  num_batch*batch_size*16(pred_len)*1        

            Y_HI_outs = Y_HI_outs.reshape(-1, Y_HI_outs.shape[-2])
            Y_HIs = Y_HIs.reshape(-1, Y_HIs.shape[-2])
            Y_SOH_outs = Y_SOH_outs.reshape(-1, Y_SOH_outs.shape[-1])
            Y_SOHs = Y_SOHs.reshape(-1,1).squeeze()   #Y_SOH(n_sample,)  Y_SOH_outs：(n_sample,n_class)
            #shape   RUL num_units*1    SOH num_units*3   HI  num_units*16(pred_len)
            Y_RUL_outs = Y_RUL_outs.reshape(-1, Y_RUL_outs.shape[-1])
            Y_RULs = Y_RULs.reshape(-1, Y_RULs.shape[-1])
            indexss = indexss.reshape(-1, indexss.shape[-1])
            indexss = torch.tensor(indexss,dtype=torch.int64)


            # result save
            folder_path = self.args.root_path + 'results/{}_{}/'.format(self.args.train_dataset_name,self.args.test_dataset_name) + setting +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


            mae_hi, mse_hi, rmse_hi, mape_hi, mspe_hi = metric(Y_HI_outs, Y_HIs)
            #soh_ce = F.cross_entropy(torch.from_numpy(Y_SOH_outs), torch.from_numpy(Y_SOHs))
            soh_lossfun = nn.NLLLoss() 
            soh_nll = soh_lossfun(torch.tensor(Y_SOH_outs), torch.LongTensor(Y_SOHs))


            df_RUL_results, mse_rul, mae_rul, r2_rul, score_rul, prec_rul, rmse_rul = RUL_results_save(self.args,Y_RUL_outs,Y_RULs,test_dataset,flag,setting,indexss)
               
            
            if flag == 'test_window':   
                Draw_RUL_decreasing_fig(self.args,df_RUL_results,score_rul,rmse_rul,prec_rul,test_dataset,flag,indexss)
            elif flag == 'test_whole':   
                Draw_RUL_unit_fig(self.args,df_RUL_results,score_rul,rmse_rul,prec_rul,test_dataset,flag,indexss)
            

            print('***************************{}_{}*******************************'.format(flag,test_dataset))
            print('train datasets:',self.args.train_dataset_name)
            print('test datasets:',self.args.test_dataset_name)
            print('op factors:',self.args.OP_features)
            print('score_rul:{0:.2f},     rmse_rul:{1:.4f}'.format(score_rul, rmse_rul))
            print('mse_hi:   {0:.4f},   mse_hi:{1:.4f}'.format(mse_hi, mae_hi))
            #print('sco_cross entropy:{0:.4f}'.format(soh_ce))
            print('soh_nll:{0:.4f}'.format(soh_nll))
            print('************************************************************')


            #perception
            if self.args.is_perception == True:
                Y_HI_outs1 = Y_HI_outs[:,self.args.label_len:,:]
                Y_HIs1 = Y_HIs[:,self.args.label_len:,:]
                mae1, mse1, rmse1, mape1, mspe1 = metric(Y_HI_outs1, Y_HIs1)

                Y_HI_outs2 = Y_HI_outs[:,-self.args.seq_len:,:]
                Y_HIs2 = Y_HIs[:,-self.args.seq_len:,:]
                mae2, mse2, rmse2, mape2, mspe2 = metric(Y_HI_outs2, Y_HIs2)

                print('perception mse:{}, mae:{}'.format(mse1, mae1))
                print('pred mse:{}, mae:{}'.format(mse2, mae2))  
            
            #csv results      
            Y_HI_outs = Y_HI_outs.reshape(Y_HI_outs.shape[0],-1)  #num_units*pred_len  100*6
            Y_HIs = Y_HIs.reshape(Y_HIs.shape[0],-1)  #num_units*pred_len  100*6

            df_index = pd.DataFrame(data=indexss,columns=['dataset_id','unit_id'])    
            #df_index['unit_id'].astype('int')
            df_hi_outs = pd.DataFrame(Y_HI_outs,columns=['pred {}'.format(i+1) for i in range(Y_HI_outs.shape[1])])
            df_his = pd.DataFrame(Y_HIs,columns=['true {}'.format(i+1) for i in range(Y_HIs.shape[1])])
            df_his = pd.concat([df_index,df_his],axis=1)
            df_hi_outs = pd.concat([df_index,df_hi_outs],axis=1)
            df_hi_results = pd.concat([df_his,df_hi_outs],axis=1)
            #df_hi_results.to_csv(folder_path +'{}_{}_HI_MAE{}_MSE{}.csv'.format(test_dataset,flag,mae_hi,mse_hi),header=True,index=False)
            
            #save soh
            results_soh = pd.DataFrame([])
            Y_SOH_outs_onehot = self._props_to_onehot(Y_SOH_outs)
            onehot = LabelBinarizer()
            onehot.fit([0,1,2])   #只有012这三类
            Y_SOHs_onehot = onehot.transform(Y_SOHs)
            Y_SOH_outs_label = onehot.inverse_transform(Y_SOH_outs_onehot)    
            pre, rec, f1, sup = precision_recall_fscore_support(Y_SOHs, Y_SOH_outs_label)
            print(classification_report(Y_SOHs, Y_SOH_outs_label))
            print("precision:", pre, "\nrecall:", rec, "\nf1-score:", f1, "\nsupport:", sup)

            results_soh['true SOH'] =  Y_SOHs
            results_soh = results_soh.astype({'true SOH':'int'})
            results_soh['assess SOH'] = Y_SOH_outs_label
            results_soh['aSOH 0'] = Y_SOH_outs[:,0]
            results_soh['aSOH 1'] = Y_SOH_outs[:,1]
            results_soh['aSOH 2'] = Y_SOH_outs[:,2]

            df_index = pd.DataFrame(data=indexss,columns=['dataset_id','unit_id'])    
            results_soh = pd.concat([df_index,results_soh],axis=1)

            
            #SOH_confusion_matrix_fig(self.args, Y_SOHs,Y_SOH_outs_label,soh_nll,test_dataset,flag)
            
            #save npy
            np.save(folder_path+'{}_{}_hi.npy'.format(test_dataset,flag),np.array([mae_hi,mse_hi,rmse_hi,mape_hi,mspe_hi]))            
            np.save(folder_path+'{}_{}_soh.npy'.format(test_dataset,flag),np.array([soh_nll,np.mean(pre),np.mean(rec),np.mean(f1)]))
            #results_soh.to_csv(folder_path + '{}_{}_SOH_NLL{}_Precesion{}.csv'.format(test_dataset,flag,soh_nll,np.mean(pre)),header=True,index=False)
            if flag == 'test_window':
                df_hi_results.to_csv(folder_path+'{}_HI.csv'.format(test_dataset),header=True,index=False)
                results_soh.to_csv(folder_path+'{}_SOH.csv'.format(test_dataset),header=True,index=False)
                
            np.save(folder_path+'{}_{}_rul.npy'.format(test_dataset,flag),np.array([score_rul,rmse_rul,prec_rul,mse_rul,mae_rul,r2_rul]))
            df_RUL_results.to_csv(folder_path+'{}_{}_RUL.csv'.format(flag,test_dataset),index=True,header=True)
            
            #if self.args.output_attention:
                #np.save(folder_path + '{}_{}_attn_weights.npy'.format(test_dataset,flag), torch.tensor(attn_weights))
                #np.save(folder_path + '{}_{}_attn_weights.npy'.format(test_dataset,flag), torch.tensor([item.cpu().detach().numpy() for item in attn_weights]).cuda())

            #return
    

    
    def predict(self, setting, load=False):
        pass
            

    def _process_one_batch_MTL(self, dataset_object, batch_x, batch_y):
        
        if self.args.model=='informer' or self.args.model=='informerstack':
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            
            batch_x = batch_x.float()[:,:,2:-3]   #delete HI and RUL column, batch_x type tensor ['HI','RUL','SOH 0','SOH 1','SOH 2']

            # decoder input
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            dec_inp = dec_inp.float()[:,:,2:-3]   #delete HI and RUL column, dec_inp is the input of decoder
  
            # encoder - decoder
            if self.args.use_amp==True:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention==True:
                        Y_HI_out, Y_SOH_out, Y_RUL_out, attn_weights = self.model(batch_x,  dec_inp)
                    elif self.args.output_attention==False:
                        Y_HI_out, Y_SOH_out, Y_RUL_out = self.model(batch_x,  dec_inp)
            elif self.args.use_amp==False:               
                if self.args.output_attention==True:
                    Y_HI_out, Y_SOH_out, Y_RUL_out, attn_weights = self.model(batch_x,  dec_inp)
                elif self.args.output_attention==False:
                    Y_HI_out, Y_SOH_out, Y_RUL_out = self.model(batch_x,  dec_inp)
            if self.args.inverse==True:
                Y_HI_out = dataset_object.inverse_transform(Y_HI_out)


            if self.args.features =='MS' or self.args.features =='S':

                Y_HI = batch_y[:,-self.args.pred_len:,-3:-2].to(self.device)     #only save the last column:HI and RUL and SOH  

            elif self.args.features =='M':
                Y_HI = batch_y[:,-self.args.pred_len:,2:-5].to(self.device)     #sensor columns, delete HI and RUL and SOH
            
            ##[sensors, HI, RUL, SOH]
            #shape       Y_SOH: (bs,)           Y_RUL: bs*1
            #shape out   Y_SOH_out: bs*3(prob)  Y_RUL: bs*1
            Y_RUL = batch_y[:,-(self.args.pred_len+1),-2:-1].to(self.device)
            Y_SOH  = batch_y[:,-(self.args.pred_len+1),-1].to(self.device).long()    
            indexs = batch_y[:,-(self.args.pred_len+1),:2] 

            if self.args.output_attention==True:
                return  Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, attn_weights, indexs
            elif self.args.output_attention==False:
                return  Y_HI_out, Y_HI, Y_RUL_out, Y_RUL, Y_SOH_out, Y_SOH, indexs
    

