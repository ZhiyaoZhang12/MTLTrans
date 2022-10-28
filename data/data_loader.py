# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelBinarizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import os
from sklearn.utils import shuffle
from torch.utils.data import Dataset

class DataPrepare(object):
    
    def __init__(self,
                  root_path,
                  pred_len,
                  seq_len,
                  HI_labeling_style,
                  dataset_name, 
                  test_data,
                  sensor_features,
                  OP_features,
                  normal_style,
                  fault_start,
                  **kwargs):
        
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.test_data = test_data
        self.sensor_features = sensor_features
        self.OP_features = OP_features
        self.normal_style = normal_style
        self.HI_labeling_style = HI_labeling_style
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.type_1 = ['FD001', 'FD003']
        self.type_2 = ['FD002', 'FD004','PHM08']
        self.fault_start = fault_start
        
    
    # data load
    def loader_engine(self, **kwargs):
        self.train = pd.read_csv(self.root_path+'data/{}/train_{}.csv'.format(self.dataset_name,self.dataset_name),header=None, **kwargs)
        if self.test_data == 'test_data':
            self.test = pd.read_csv( self.root_path +'data/{}/test_{}.csv'.format(self.dataset_name,self.dataset_name) ,header=None, **kwargs)
            self.test_RUL =  pd.read_csv(self.root_path +'data/{}/RUL_{}.csv'.format(self.dataset_name,self.dataset_name),header=None, **kwargs)
        elif  self.test_data == 'train_data': 
            self.test = pd.read_csv(self.root_path+'data/{}/train_{}.csv'.format(self.dataset_name,self.dataset_name),header=None, **kwargs) 

    def add_columns_name(self):
        sensor_columns = ["sensor {}".format(s) for s in range(1,22)]
        info_columns = ['unit_id','cycle']
        settings_columns = ['setting 1', 'setting 2', 'setting 3']
        self.train.columns = info_columns + settings_columns + sensor_columns
        self.test.columns = info_columns + settings_columns + sensor_columns
        if self.test_data == 'test_data':
            self.test_RUL.columns = ['RUL']
            self.test_RUL['unit_id'] = [i for i in range(1,len(self.test_RUL)+1,1)]
            self.test_RUL.set_index('unit_id',inplace=True,drop=True)
        
        
    def labeling_train(self,data, piecewise_point,falut_start):
        ##for train
        maxRUL_dict = data.groupby('unit_id')['cycle'].max().to_dict()
        data['maxRUL'] = data['unit_id'].map(maxRUL_dict)
        data['RUL'] = data['maxRUL'] - data['cycle']
        
        #linear
        data['HI_linear'] = 1 - data['cycle']/data['maxRUL']
        #piece_wise linear
        FPT = data['maxRUL'] - piecewise_point
        data['HI_pw_linear'] =  data['cycle']/ (FPT - data['maxRUL'] ) + data['maxRUL']/(data['maxRUL'] - FPT)
        filter_piece_wise = (data['cycle'] <= FPT)
        data.loc[filter_piece_wise,['HI_pw_linear']] = 1
        #quadratic
        data['HI_quadratic'] = 1 -((data['cycle']*data['cycle'])/(data['maxRUL']*data['maxRUL']))
        #piece_wise quadratic
        data['HI_pw_quadratic'] = 1 - ((1/(piecewise_point**2))*(data['cycle']-FPT)**2)
        filter_piece_wise = (data['cycle'] <= FPT)
        data.loc[filter_piece_wise,['HI_pw_quadratic']] = 1
        
        data['HI'] = data[self.HI_labeling_style]
        data.drop(['HI_linear','HI_pw_linear','HI_quadratic','HI_pw_quadratic'],axis=1,inplace=True)      
        
        #RUL filter
        filter_RUL = (data['RUL'] >= piecewise_point)
        data.loc[filter_RUL,['RUL']] = piecewise_point
            
        #SOH labeling
        data['SOH'] = data['HI'].values
        filter_health = (data['HI']>=1)
        data.loc[filter_health,['SOH']] = 2
        filter_degra = (data['HI']<1) & (data['HI']>=falut_start)
        data.loc[filter_degra,['SOH']] = 1
        filter_fault = (data['HI']<falut_start)
        data.loc[filter_fault,['SOH']] = 0
        return data
        

    def labeling_test(self,data,piecewise_point,falut_start):
        ###for test
        RUL_dict = self.test_RUL.to_dict()
        data['RUL_test'] =data['unit_id'].map(RUL_dict['RUL'])

        maxT_dict_train = data.groupby('unit_id')['cycle'].max().to_dict()
        data['maxT'] = data['unit_id'].map(maxT_dict_train)

        data['RUL'] = data['RUL_test'] + data['maxT'] - data['cycle']
        max_RUL_test = data.groupby('unit_id')['RUL'].max().to_dict()
        data['maxRUL'] = data['unit_id'].map(max_RUL_test)
    
        #linear
        data['HI_linear'] = 1 - data['cycle']/data['maxRUL']
        #piece_wise linear
        FPT = data['maxRUL'] - piecewise_point + 1 
        data['HI_pw_linear'] =  data['cycle']/ (FPT - data['maxRUL'] ) + data['maxRUL']/(data['maxRUL'] - FPT)
        filter_piece_wise = (data['cycle'] <= FPT)
        data.loc[filter_piece_wise,['HI_pw_linear']] = 1
        #quadratic
        data['HI_quadratic'] = 1 -((data['cycle']*data['cycle'])/(data['maxRUL']**2))
        #piece_wise quadratic
        data['HI_pw_quadratic'] = 1 -(((data['cycle']-FPT)**2)/(piecewise_point**2))
        filter_piece_wise = (data['cycle'] <= FPT)
        data.loc[filter_piece_wise,['HI_pw_quadratic']] = 1
        
        data['HI'] = data[self.HI_labeling_style]
        #'maxRUL'还有用，暂时不要删除
        data.drop(['RUL_test','maxT','HI_linear','HI_pw_linear','HI_quadratic','HI_pw_quadratic'],axis=1,inplace=True)
                
        #RUL filter
        filter_RUL = (data['RUL'] >= piecewise_point)
        data.loc[filter_RUL,['RUL']] = piecewise_point
        
        #SOH labeling
        data['SOH'] = data['HI'].values
        filter_health = (data['HI']>=1)
        data.loc[filter_health,['SOH']] = 2
        filter_degra = (data['HI']<1) & (data['HI']>=falut_start)
        data.loc[filter_degra,['SOH']] = 1
        filter_fault = (data['HI']<falut_start)
        data.loc[filter_fault,['SOH']] = 0
        return data


    def onehot_coding_SOH(self):
        onehot = LabelBinarizer()
        train_soh = onehot.fit_transform(self.train['SOH'])
        test_soh = onehot.fit_transform(self.test['SOH'])
        
        soh_columns = ["SOH {}".format(s) for s in range(3)]
        self.train[soh_columns] = train_soh
        self.test[soh_columns] = test_soh
            
        self.train.drop(['SOH'],inplace=True,axis=1)
        self.test.drop(['SOH'],inplace=True,axis=1)
        return onehot
      
    
    def normalization(self):
        df_train = self.train
        df_test = self.test

        if len(df_train):
            df_train_normalize = self.train.copy()
            df_test_normalize = self.test.copy()

        if self.dataset_name in self.type_1:
            if self.normal_style == 'StandardScaler':
                scaler = StandardScaler().fit(df_train[self.sensor_features])
            elif self.normal_style == 'MinMaxScaler':
                scaler = MinMaxScaler().fit(df_train[self.sensor_features])

            df_train_normalize[self.sensor_features] = scaler.transform(df_train[self.sensor_features])
            df_test_normalize[self.sensor_features] = scaler.transform(df_test[self.sensor_features])

        elif self.dataset_name in self.type_2:   
            #给他们聚类['OP']
            self.settings_columns = ['setting 1', 'setting 2', 'setting 3']
            kmeans = KMeans(n_clusters=6, random_state=0).fit(df_train[self.settings_columns])
            df_train['OP'] = kmeans.labels_
            df_test['OP'] = kmeans.predict(df_test[self.settings_columns])

            if len(df_train):
                df_train_normalize = df_train.copy()
                df_test_normalize = df_test.copy()

            gb = df_train.groupby('OP')[self.sensor_features]

            d = {}
            for x in gb.groups:
                if self.normal_style == 'StandardScaler':
                    d["scaler_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))
                elif self.normal_style == 'MinMaxScaler':
                    d["scaler_{0}".format(x)] = MinMaxScaler().fit(gb.get_group(x))

                df_train_normalize.loc[df_train_normalize['OP'] == x, self.sensor_features] = d["scaler_{0}".format(x)].transform(
                    df_train.loc[df_train['OP'] == x, self.sensor_features])
                df_test_normalize.loc[df_test_normalize['OP'] == x, self.sensor_features] = d["scaler_{0}".format(x)].transform(
                    df_test.loc[df_test['OP'] == x, self.sensor_features])

        self.train = df_train_normalize.copy()
        self.test = df_test_normalize.copy()
        del df_train_normalize, df_test_normalize
        
    def onehot_coding_OP(self):
        if self.dataset_name in self.type_2:
            onehot = LabelBinarizer()
            train_op = onehot.fit_transform(self.train['OP'])
            test_op = onehot.fit_transform(self.test['OP'])

            self.op_columns = ["OP {}".format(s) for s in range(1,7)]
            self.train[self.op_columns] = train_op
            self.test[self.op_columns] = test_op

            self.train.drop(['OP'],inplace=True,axis=1)
            self.test.drop(['OP'],inplace=True,axis=1)
        
        else: 
            #pass
            self.op_columns = ["OP {}".format(s) for s in range(1,7)]
            train_op = np.zeros((len(self.train),6))
            test_op = np.zeros((len(self.test),6))
            self.train[self.op_columns] = train_op
            self.test[self.op_columns] = test_op
            
        
    def del_unuseful_columns(self): 
        #if self.dataset_name in self.type_2:
        if self.OP_features == True:
            useful_columns =  ['unit_id', 'cycle','maxRUL'] + self.sensor_features + self.op_columns + ['HI','RUL','SOH']  
        #elif self.dataset_name in self.type_1:
        if self.OP_features == False:
            useful_columns =  ['unit_id', 'cycle','maxRUL'] + self.sensor_features + ['HI','RUL','SOH']
        
        self.train = self.train.loc[:,useful_columns]     
        self.test = self.test.loc[:,useful_columns] 
        
        #以便后续做双索引
        self.train['dataset_id'] = ['{}'.format(self.dataset_name)]*self.train.shape[0]
        self.test['dataset_id'] = ['{}'.format(self.dataset_name)]*self.test.shape[0]
        
        
    def process(self):
        self.loader_engine()
        self.add_columns_name()
       
        self.train = self.labeling_train(self.train,piecewise_point=125,falut_start=self.fault_start)
        if self.test_data == 'test_data':
            self.test = self.labeling_test(self.test,piecewise_point=125,falut_start=self.fault_start)
        elif self.test_data == 'train_data':  
            self.test = self.labeling_train(self.test,piecewise_point=125,falut_start=self.fault_start)
            
        self.train = self.train.astype({'RUL':np.int64,'SOH':np.int64})
        self.test = self.test.astype({'RUL':np.int64,'SOH':np.int64})  
        self.normalization()
        if self.OP_features == True:
            self.onehot_coding_OP()
        self.del_unuseful_columns()
        self.train.to_csv(self.root_path + 'data/{}/train.csv'.format(self.dataset_name),index=False)
        self.test.to_csv(self.root_path + 'data/{}/test.csv'.format(self.dataset_name),index=False)
        
        
class TrainValiTest(Dataset): 
    def __init__(self, root_path,train_dataset_name,test_dataset_name,
                 seq_len,pred_len,label_len,
                 sensor_features, is_padding, data_augmentation,
                 is_descrsing, normal_style, synthetic_data_path, 
                 HI_labeling_style):

        self.HI_labeling_style = HI_labeling_style
        self.sensor_features = sensor_features
        
        # info
        self.root_path = root_path
        
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.label_len = label_len
        
        self.is_padding = is_padding
        self.data_augmentation = data_augmentation
        self.is_descrsing = is_descrsing
        self.normal_style = normal_style
        self.synthetic_data_path = synthetic_data_path

              
    ###padding
    def back_padding_RtF(self,data):
        '''
        1.pad a zero-matric (pred_len*NO.sensors) in the end of each trajectory;    
        2.train_test split then normalization，then padding;
        '''
        data_new = pd.DataFrame([],columns=data.columns) #for saving new data         

        for unit_index in data.index.to_series().unique():
            unit_df = pd.DataFrame(data.loc[unit_index])    #trajectory of each unit            
            padding_data = pd.DataFrame(data=np.full(shape=[self.pred_len,data.shape[1]],fill_value=0),columns=data.columns)

            #pad
            temp_new = pd.concat([unit_df,padding_data])
            temp_new['cycle'] = [i for i in range(1,len(temp_new)+1)]
            temp_new.index = [unit_index]*len(temp_new)
            temp_new['maxRUL'] = [unit_df['maxRUL'].max()]*(len(temp_new)) 
            temp_new['unit_id'] = [unit_df.index[0][1]]*len(temp_new)
            temp_new['dataset_id'] = [unit_df.index[0][0]]*len(temp_new)   
            
            data_new = pd.concat((data_new,temp_new),axis=0)

        data_new.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
        return data_new

    def HI_UtD(self,data,piecewise_point):
        '''
        test data is UtD, so construct the predicted PDIs
        1.back_padding for test data, then modified the PDIs
        '''   
        data_new = pd.DataFrame([],columns=data.columns)  #for saving new data   

        for unit_index in data.index.to_series().unique():
            unit_df = pd.DataFrame(data.loc[unit_index])    #trajectory of each unit  

            padding_data = pd.DataFrame(data=np.full(shape=[self.pred_len,data.shape[1]],fill_value=0),columns=data.columns) #use zero-matrics first
            temp_new = pd.concat([unit_df,padding_data])
            temp_new['cycle'] = [(i+unit_df['cycle'].min()) for i in range(0,len(temp_new))]
            temp_new['maxRUL'] = [unit_df['maxRUL'].max()]*(len(temp_new))
            temp_new['unit_id'] = [unit_index[1]]*len(temp_new)
            temp_new['dataset_id'] = [unit_index[0]]*len(temp_new)

            #重新算HI
            FPT = temp_new['maxRUL'] - piecewise_point  + 1                                    
            if self.HI_labeling_style == 'HI_linear':
                temp_new['HI'] = 1 - temp_new['cycle']/temp_new['maxRUL']
            elif self.HI_labeling_style == 'HI_pw_linear':
                temp_new['HI'] =  temp_new['cycle']/ (FPT - temp_new['maxRUL'] ) + temp_new['maxRUL']/(temp_new['maxRUL'] - FPT)
                filter_piece_wise = (temp_new['cycle'] <= FPT)
                temp_new.loc[filter_piece_wise,['HI']] = 1
            elif self.HI_labeling_style == 'HI_quadratic':
                temp_new['HI'] = 1 -((temp_new['cycle']*temp_new['cycle'])/(temp_new['maxRUL']**2))
            elif self.HI_labeling_style == 'HI_pw_quadratic':
                temp_new['HI'] = 1 -(((temp_new['cycle']-FPT)**2)/(piecewise_point**2))
                filter_piece_wise = (temp_new['cycle'] <= FPT)
                temp_new.loc[filter_piece_wise,['HI']] = 1

            #if the constructed PDIs is less than 0,  use 0 instead
            filter_neg_value = (temp_new['HI'] < 0)
            temp_new.loc[filter_neg_value,['HI']] = 0            
            temp_new.drop('maxRUL',inplace=True,axis=1)

            #concat
            data_new = pd.concat([data_new,temp_new],axis=0)

        data_new.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
        return data_new
    
  

    def spilt_data(self):
        
        path_save = self.root_path + 'data/{}_{}/'.format(self.train_dataset_name,self.test_dataset_name)   
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        
        # load data            
        df_train = pd.DataFrame([])
        for dataset in self.train_dataset_name:  
            train_data = pd.read_csv(os.path.join(self.root_path,'data/{}/train.csv'.format(dataset)),header=0,index_col=['dataset_id','unit_id'])
            df_train = pd.concat([df_train,train_data],axis=0)  
            
        train_turbines = np.arange(len(df_train.index.to_series().unique()))
        train_turbines, validation_turbines = train_test_split(train_turbines, test_size=0.3,random_state = 1334) 
        idx_train = df_train.index.to_series().unique()[train_turbines]
        idx_validation = df_train.index.to_series().unique()[validation_turbines]     

        train = df_train.loc[idx_train]
        validation = df_train.loc[idx_validation]

        #padding
        if self.is_padding==True:
            train = self.back_padding_RtF(train)
            validation = self.back_padding_RtF(validation)

        train.drop(['maxRUL','cycle'],inplace=True,axis=1)
        validation.drop(['maxRUL','cycle'],inplace=True,axis=1)

        train.to_csv(path_save+'{}_train_normal.csv'.format(self.train_dataset_name),header=True,index=True)
        validation.to_csv(path_save+'{}_validation_normal.csv'.format(self.train_dataset_name),header=True,index=True)
           
        ## test
        #test = pd.DataFrame([])
        for dataset in self.test_dataset_name:
            #test_data = pd.read_csv(os.path.join(self.root_path,'data/{}/test.csv'.format(dataset)),header=0,index_col=['dataset_id','unit_id'])
            #test = pd.concat([test,test_data],axis=0)
            test = pd.read_csv(os.path.join(self.root_path,'data/{}/test.csv'.format(dataset)),header=0,index_col=['dataset_id','unit_id'])
            
            #pading  
            test = self.HI_UtD(test,piecewise_point=125) 
            test.drop(['maxRUL','cycle'],inplace=True,axis=1)
            
            ###test get the last window
            test_window = pd.DataFrame([])
            for unit_index in (test.index.to_series().unique()):
                trajectory_df = pd.DataFrame(test.loc[unit_index])

                if len(trajectory_df) >= (self.seq_len +self.pred_len) :
                    temp_last_new = trajectory_df.iloc[(-self.seq_len-self.pred_len):,:]  
                    test_window = pd.concat([test_window,temp_last_new])             
                else:                
                    padding_data = pd.DataFrame(data=np.full(shape=[-len(trajectory_df)+self.seq_len+self.pred_len,trajectory_df.shape[1]],fill_value=1),columns=trajectory_df.columns)
                    temp_last_new = pd.concat([padding_data,trajectory_df])
                    temp_last_new['unit_id'] = [unit_index[1]]*len(temp_last_new)
                    temp_last_new['dataset_id'] = [unit_index[0]]*len(temp_last_new)

                    temp_last_new.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
                    test_window = pd.concat([test_window,temp_last_new])

            test_window.to_csv(path_save+'{}_test_window_normal.csv'.format(dataset),header=True,index=True)   
            test.to_csv(path_save+'{}_test_whole_normal.csv'.format(dataset),header=True,index=True)   

        

class DataReaderTrajactory(Dataset):
    
    def __init__(self, 
                 root_path,dataset_name,train_dataset_name,test_dataset_name,
                 sensor_features, is_padding, data_augmentation,
                 is_descrsing, normal_style, synthetic_data_path, 
                 HI_labeling_style,flag='pred', size=None, 
                 features='MS',
                 target='HI', inverse=False, timeenc=0,            
                 cols=None):

        self.HI_labeling_style = HI_labeling_style
        self.sensor_features = sensor_features
        
        # info
        self.root_path = root_path

        # init
        assert flag in ['train', 'test_window','test_whole', 'val']
        type_map = {'train':0, 'val':1, 'test_whole':2,'test_window':3}
        self.set_type = type_map[flag]
        self.flag = flag
        self.inverse = inverse
        
        self.data_x = pd.DataFrame(data=[])
        self.data_y = pd.DataFrame(data=[])
        self.all_seq_x = []
        self._all_seq_y = []

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.dataset_name = dataset_name
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        
        self.features = features
        self.target = target
        self.inverse = inverse
         
        self.is_padding = is_padding
        self.data_augmentation = data_augmentation
        self.is_descrsing = is_descrsing
        self.normal_style = normal_style
        self.synthetic_data_path = synthetic_data_path

        self.__read_data__()  # data preprocessing
              

    def __read_data__(self):
          
        '''
        1. spilt for data,val,test according to trajectory
        2. normalization
        3. prepare the data for encoder and decoder according to trajectory  96*14； 72*14  (self.transform_data)
        '''
        data_path = self.root_path + 'data/{}_{}/'.format(self.train_dataset_name,self.test_dataset_name)
        if self.flag =='train':
            #self.data_x = train
            self.data_x = pd.DataFrame([])
            for dataset in self.train_dataset_name:
                data =  pd.read_csv(data_path + '{}_train_normal.csv'.format(self.dataset_name),header=0,index_col=['dataset_id','unit_id'])
                self.data_x = pd.concat([self.data_x,data],axis=0)
            
        elif self.flag =='val':
            #self.data_x = validation
            self.data_x = pd.DataFrame([])
            for dataset in self.train_dataset_name:
                data =  pd.read_csv(data_path + '{}_validation_normal.csv'.format(self.dataset_name),header=0,index_col=['dataset_id','unit_id'])
                self.data_x = pd.concat([self.data_x,data],axis=0)
                
        elif self.flag =='test_window':
            self.data_x = pd.read_csv(data_path + '{}_test_window_normal.csv'.format(self.dataset_name),header=0,index_col=['dataset_id','unit_id'])
            #self.data_x = test
        elif self.flag =='test_whole':
            self.data_x = pd.read_csv(data_path + '{}_test_whole_normal.csv'.format(self.dataset_name),header=0,index_col=['dataset_id','unit_id'])
            #self.data_x = test_window
                
            
        if self.inverse:
            self.data_y = self.data_x
        else:
            self.data_y = self.data_x
            
        ## prepare the data for encoder and decoder according to trajectory:
        self.all_seq_x, self.all_seq_y = self.transform_data()

        
    def transform_data(self):
        ### enc, dec for save the precessed data(time window) 
        enc,dec = [],[]
        print('\n')
        print('There are {} trajectories in {} {} dataset'.format(len(self.data_x.index.to_series().unique()),self.dataset_name,self.flag))
        
        #复制这两列，再将其移动到前两列    
        self.data_x.reset_index(inplace=True,drop=False)
        
        #改写dataset_id  因为batch_x里面不能有str?
        self.data_x['dataset'] = self.data_x['dataset_id'] 
        filter1 = self.data_x['dataset_id'] == 'FD001'
        self.data_x.loc[filter1,'dataset'] = 1
        filter2 = self.data_x['dataset_id'] == 'FD002'
        self.data_x.loc[filter2,'dataset'] = 2
        filter3 =  self.data_x['dataset_id'] == 'FD003'
        self.data_x.loc[filter3,'dataset'] = 3
        filter4 = self.data_x['dataset_id'] == 'FD004'
        self.data_x.loc[filter4,'dataset'] = 4
        filter8 = self.data_x['dataset_id'] == 'PHM08'
        self.data_x.loc[filter8,'dataset'] = 8

        self.data_x['unit'] = self.data_x['unit_id']
        self.data_x.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
        orders = [-2,-1] + [i for i in range(len(self.data_x.columns)-2)]
        self.data_x = self.data_x.iloc[:,orders]

        #Loop through each trajectory
        for unit_index in (self.data_x.index.to_series().unique()): 
            #get the whole trajectory (index)
            temp_df = pd.DataFrame(self.data_x.loc[unit_index])             
             
            # Loop through the data in the object (index) trajectory
            data_enc_npc, data_dec_npc, array_data_enc, array_data_dec = [],[],[],[]
            len_trajectory = len(temp_df)
            
            enc_last_index = len_trajectory - self.pred_len
            
            for i in range(enc_last_index - self.seq_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len 
                r_end = r_begin + self.label_len + self.pred_len 

                data_enc_npc = temp_df.iloc[s_begin:s_end]
                data_dec_npc = temp_df.iloc[r_begin:r_end]
       
                array_data_enc.append(data_enc_npc)
                array_data_dec.append(data_dec_npc)
        
            enc = enc + array_data_enc
            dec = dec + array_data_dec

        return enc,dec
    
            
    def __getitem__(self,index):  
        
        seq_x = self.all_seq_x[index].values
        seq_y = self.all_seq_y[index].values
        
        return seq_x.astype(np.float32),seq_y.astype(np.float32)
        
        
    def __len__(self):
        
        return len(self.all_seq_x)   
    
    def inverse_transform(self, data):
        
        return self.scaler.inverse_transform(data)
