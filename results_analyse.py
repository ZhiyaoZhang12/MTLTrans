import pandas as pd
import numpy as np

def results_analysis_MTL(args,setting_analysis_name,test_dataset,flag):

    all_df_metrics_hi = pd.DataFrame()
    all_df_metrics_rul = pd.DataFrame()
    all_df_metrics_soh = pd.DataFrame()

    for k in range(args.itr):  #args.itr
        file_name = setting_analysis_name + '{}'.format(k)

        metrics_hi = np.load(args.root_path + 'results/{}_{}/{}/{}_{}_hi.npy'.format(args.train_dataset_name,args.test_dataset_name,file_name,test_dataset,flag))
        metrics_rul = np.load(args.root_path + 'results/{}_{}/{}/{}_{}_rul.npy'.format(args.train_dataset_name,args.test_dataset_name,file_name,test_dataset,flag))
        metrics_soh = np.load(args.root_path + 'results/{}_{}/{}/{}_{}_soh.npy'.format(args.train_dataset_name,args.test_dataset_name,file_name,test_dataset,flag))


        df_metrics_hi = pd.DataFrame(data = [metrics_hi], columns = ['mae', 'mse', 'rmse', 'mape', 'mspe']) 
        df_metrics_rul = pd.DataFrame(data = [metrics_rul], columns = ['score', 'rmse', 'prec', 'mse', 'mae', 'r2' ]) 
        df_metrics_soh = pd.DataFrame(data = [metrics_soh], columns = ['NLLoss', 'precision', 'recall', 'f1']) 

        all_df_metrics_hi = all_df_metrics_hi.append(df_metrics_hi)
        all_df_metrics_rul = all_df_metrics_rul.append(df_metrics_rul)
        all_df_metrics_soh = all_df_metrics_soh.append(df_metrics_soh)


    means_hi = pd.DataFrame([all_df_metrics_hi.mean()],index=['mean'])
    means_rul = pd.DataFrame([all_df_metrics_rul.mean()],index=['mean'])
    means_soh = pd.DataFrame([all_df_metrics_soh.mean()],index=['mean'])

    all_df_metrics_hi = all_df_metrics_hi.append(means_hi)
    all_df_metrics_rul = all_df_metrics_rul.append(means_rul)
    all_df_metrics_soh = all_df_metrics_soh.append(means_soh)

    all_df_metrics_hi['HI'] = ['hi']*(len(all_df_metrics_hi)-1) + ['mean']
    all_df_metrics_rul['RUL'] = ['rul']*(len(all_df_metrics_rul)-1) + ['mean']
    all_df_metrics_soh['SOH'] = ['soh']*(len(all_df_metrics_soh)-1) + ['mean']
    
    all_df_metrics_hi.drop(['HI','rmse', 'mape', 'mspe'],axis=1,inplace=True)  
    all_df_metrics_rul.drop(['RUL','mse', 'mae', 'r2'],axis=1,inplace=True)
    all_df_metrics_soh.drop(['SOH'],axis=1,inplace=True)

    print('-------------mean_hi_{}----------------'.format(flag))
    print(all_df_metrics_hi.iloc[-1,:]) 
    print('-------------mean_rul_{}----------------'.format(flag))
    print(all_df_metrics_rul.iloc[-1,:])
    print('-------------mean_soh_{}----------------'.format(flag))
    print(all_df_metrics_soh.iloc[-1,:])

    return all_df_metrics_hi, all_df_metrics_rul, all_df_metrics_soh

def results_analysis_oRUL(args,setting_analysis_name,test_dataset,flag):
    all_df_metrics_rul = pd.DataFrame()

    for k in range(args.itr):  #args.itr
        file_name = setting_analysis_name + '{}'.format(k)

        metrics_rul = np.load(args.root_path + 'results/{}_{}/{}/{}_{}_rul.npy'.format(args.train_dataset_name,args.test_dataset_name,file_name,test_dataset,flag))

        df_metrics_rul = pd.DataFrame(data = [metrics_rul], columns = ['score', 'rmse', 'prec', 'mse', 'mae', 'r2' ]) 
        all_df_metrics_rul = all_df_metrics_rul.append(df_metrics_rul)

    means_rul = pd.DataFrame([all_df_metrics_rul.mean()],index=['mean'])
    all_df_metrics_rul = all_df_metrics_rul.append(means_rul)
    all_df_metrics_rul['RUL'] = ['rul']*(len(all_df_metrics_rul)-1) + ['mean']
    all_df_metrics_rul.drop(['RUL','mse', 'mae', 'r2'],axis=1,inplace=True)

    print('-------------mean_rul_{}----------------'.format(flag))
    print(all_df_metrics_rul.iloc[-1,:])

    return  all_df_metrics_rul