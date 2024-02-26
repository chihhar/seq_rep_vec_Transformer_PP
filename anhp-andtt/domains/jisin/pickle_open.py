import pickle
import numpy as np
import pandas as pd
import pdb
import torch
#保存したデータフレームの呼び出し
def rolling_matrix(x,time_step):
            x = x.flatten()
            n = x.shape[0]
            stride = x.strides[0]
            return np.lib.stride_tricks.as_strided(x, shape=(n-time_step+1, time_step), strides=(stride,stride) ).copy()
def generate_data(file_name,dir_name,split=[0.8,0.1,0.1]):
    time_step=30
    print(f"time_step:{time_step}")
    train_per,valid_per,test_per=split
    df = pd.read_csv(f"{dir_name}{file_name}")
    df["DateTime"]=pd.to_datetime(df["DateTime"])
    
    df["time_since_start"]=df["DateTime"].map(pd.Timestamp.timestamp)/3600
    save_df=df[["time_since_start"]][1:]
    save_df["time_since_last_event"]=np.ediff1d(df["time_since_start"])
    save_df["type_event"]=1
    #df["time_since_last_event"]=df["time_since_start"][1:]-df["time_since_start"][:-1]
    df_len=len(save_df)
    #save_df["name"]="unknown(unknown)"
    #save_df["name"][0]="bos"
    #save_df["name"][df_len-1]="eos"
    #save_df = save_df.loc[:, ["name","time"]]
    #pdb.set_trace()
    #save_df=np.array(save_df)
    #save_df=save_df.to_dict(orient='index')
    #save_df=pd.DataFrame((df["DateTime"].map(pd.Timestamp.timestamp)/3600).values,columns="times")
    
    all_df=save_df.to_dict()#np.array(save_df.to_dict(orient='records')).tolist()
    train_dict =save_df[:int(train_per*df_len)].to_dict(orient="list")# np.array().tolist()
    valid_dict = save_df[int(train_per*df_len):int((train_per+valid_per)*df_len)].to_dict(orient="list")#np.array().tolist()
    test_dict = save_df[int((train_per+valid_per)*df_len):].to_dict(orient="list")#np.array().tolist()
    #param data: list[list[dict{"time_since_last_event"[float], "time_since_start"[float], "type_event"[int]}]]
    d2 = {'dim_process': 100, 'k3': 3, 'k4': 4}
    train_np =np.array(save_df["time_since_last_event"][:int(train_per*df_len)])
    valid_np = np.array(save_df["time_since_last_event"][int(train_per*df_len):int((train_per+valid_per)*df_len)])
    test_np = np.array(save_df["time_since_last_event"][int((train_per+valid_per)*df_len):])
    
    pdb.set_trace()
    
    train_data = torch.tensor(rolling_matrix(train_np,time_step)).to(torch.double)
    train_list = train_data.tolist()
    #train_dataset = torch.utils.data.TensorDataset(train_data)
    #valid_data=df[int(len(df)*0.8):int(len(df)*0.9)]
    
    valid_data = torch.tensor(rolling_matrix(valid_np,time_step)).to(torch.double)
    valid_list = valid_data.tolist()
    
    test_data = torch.tensor(rolling_matrix(test_np,time_step)).to(torch.double)
    test_list = test_data.tolist()
    
    
    pdb.set_trace()
    with open('./jisinall.pkl','wb') as f:
        pickle.dump(train_dict,f)
    with open('./train.pkl','wb') as f:
        pickle.dump(train_list,f)
    with open('./dev.pkl','wb') as f:
        pickle.dump(valid_list,f)
    with open('./test.pkl','wb') as f:
        pickle.dump(test_list,f)

if __name__ == "__main__":
    generate_data("date_jisin.90016","./")