import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.pyplot import MultipleLocator  
from models.score_fun import score_loss, score_focal_loss
from utils.metrics import metric, error_function, score_function, prec_function, accuracy_function
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import torch
import os


def RUL_results_save(args, preds, trues, test_dataset, flag, setting,indexss): 
    preds = preds - 0.5
    preds = preds.round()
    preds[preds < 0] = 0
    
    df_test = pd.DataFrame([])
    df_index = pd.DataFrame(data=indexss,columns=['dataset_id','unit_id']) 
    df_index['unit_id'].astype('int')
    #df_index['dataset_id'] = ['FD00{}'.format(int(i))  for i in df_index['dataset_id']]
    df_index['dataset_id'] = [test_dataset] * len(df_index['dataset_id'])
    df_test = pd.concat([df_index,df_test],axis=1)
    df_test.set_index(['dataset_id','unit_id'],inplace=True,drop=True)
    
    df_test['True_RUL'] = trues.reshape(-1)
    df_test['Predicted_RUL'] = preds.reshape(-1)
    
    df_test = df_test.astype({'Predicted_RUL':'int','True_RUL':'int'})
    

    df_test['Error'] = df_test.apply(lambda df: error_function(df, 'Predicted_RUL', 'True_RUL'), axis=1)
    df_test['Score'] = df_test.apply(lambda df: score_function(df, 'Error'), axis=1)
    df_test['Prec'] = df_test.apply(lambda df: prec_function(df, 'Predicted_RUL', 'True_RUL'), axis=1)
    df_test['Accuracy'] = df_test.apply(lambda df: accuracy_function(df, 'Error'), axis=1)

    r2 = r2_score(df_test['True_RUL'], df_test['Predicted_RUL'])
    prec = df_test['Prec'].sum()/len(df_test['Prec'])
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    
    if args.test_data == 'test_data':
        score = score_loss(torch.tensor(preds), torch.tensor(trues)) 
    elif args.test_data == 'train_data':
        score = score_loss(torch.tensor(preds), torch.tensor(trues))
        score = score/len(df_test)
        
    print('[RUL performance]:  score:{}, rmse:{}, prec:{}'.format(score, rmse, prec))

    # result save
    folder_path = args.root_path + 'results/{}_{}/'.format(args.train_dataset_name,args.test_dataset_name) + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    #df_test.to_csv(folder_path + '{5}_RUL_{4}_{0}_SCORE{1:.4f}_RMSE{2:.4f}_Prec{3:.4f}.csv'.format(flag, score, rmse,prec,args.test_data,test_dataset),index=True,header=True) 
    return  df_test, mse, mae, r2, score, prec, rmse



def Draw_RUL_decreasing_fig(args, df_test, score, rmse, prec,test_dataset,flag,indexss):      
    #matplotlib.rcParams.update({'font.size': 18,'font.family' : 'Times New Roman'}) 
    #plt.figure(figsize=(8,6),dpi=600)
    plt.figure(figsize=(8,6))
    #plt.subplots_adjust(wspace=4)

    print(df_test.head(10))


    data_sort= df_test.sort_values(by='True_RUL', axis=0, ascending= False)
    #print(data_sort.head(10))

    X = range(len(data_sort))

    # x_major_locator=MultipleLocator(10) 
    y_major_locator=MultipleLocator(20)#  
    ax=plt.gca()

    # ax.xaxis.set_major_locator(x_major_locator)  
    ax.yaxis.set_major_locator(y_major_locator)

    plt.ylim(-5,145)  #
    plt.xlim(-5,len(X)+15)  



    l1=plt.plot(X,data_sort['True_RUL'],label="Ground truth RUL",color="royalblue", 
                linewidth=3,markerfacecolor='royalblue',linestyle='--')
    l2=plt.plot(X,data_sort['Predicted_RUL'],label="Predicted RUL",
                color="red",linewidth=1.7,marker='.',markerfacecolor='r',
                markersize=7,linestyle='-')

    plt.xlabel("Engines(decreasing RUL)")
    plt.ylabel("RUL/Cycle")
    plt.title("score {}".format(score))
    plt.legend(loc = 'upper right')
    
    fig_path = args.root_path + 'Fig/{}_{}/'.format(args.train_dataset_name,args.test_dataset_name)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(fig_path + "{5}_RUL_{4}_{3}_Score{0:.4f}_RMSE{1:.4f}_Prec{2:.4f}.png".format(score,rmse,prec,flag,args.test_data,test_dataset), bbox_inches='tight')
    #plt.show()
    
def Draw_RUL_unit_fig(args, df_test, score, rmse, prec,test_dataset,flag, indexss):  
   
    matplotlib.rcParams.update({'font.size': 12}) 
    plt.figure(figsize=(8,6))
    plt.subplots_adjust(wspace=0.35,hspace=0.35)
    
    data_index = df_test.index.to_series().unique()
    unit_draw = [data_index[i] for i in [0,10,20,30]]
    print(unit_draw)
    
    for i in range(len(unit_draw)):
        data_unit = df_test.loc[unit_draw[i],:]
        X = range(len(data_unit))
        
        ax = plt.subplot(2,2,i+1)
        

        from matplotlib.pyplot import MultipleLocator
        x_major_locator=MultipleLocator(30)
        y_major_locator=MultipleLocator(20)
        pax=plt.gca()
        pax.xaxis.set_major_locator(x_major_locator)
        pax.yaxis.set_major_locator(y_major_locator) 
        plt.ylim(-5,145)
        plt.xlim(-5,len(X)+10)
        
        
        l1=plt.plot(X,data_unit['True_RUL'],label="actual RUL",color="b", 
                    linewidth=3,markerfacecolor='b',linestyle='--')
        l2=plt.plot(X,data_unit['Predicted_RUL'],label="predicted RUL",
                    color="red",linewidth=1.7,marker='.',markerfacecolor='r',
                    markersize=7,linestyle='-')
        plt.xlabel("Time/Cycle")
        plt.ylabel("RUL/Cycle")
        plt.title(label='{}'.format(unit_draw[i]),loc='left')


        plt.legend(loc = 'lower left',bbox_to_anchor=(0, 0))  
      
        
    fig_path = args.root_path + 'Fig/{}_{}/'.format(args.train_dataset_name,args.test_dataset_name)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(fig_path + "{5}_RUL_{4}_{3}_Score{0:.4f}_RMSE{1:.4f}_Prec{2:.4f}.png".format(score,rmse,prec,flag,args.test_data,test_dataset), bbox_inches='tight')
    #plt.show()