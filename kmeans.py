import numpy as np
import itertools 
import pdb
import tqdm
from sklearn.cluster import KMeans
def Set_data_kmeans( input, n_clusters):
    target_data = []
    #input [[~B,L],[1B~2B,L],[~N,L]] :input_time
    #      [[~B,x,y],
    
    for batch in input:
        if len(batch)==2:
            batch=batch[0]
        if batch.shape[1]>2:
            target_data=np.append(target_data, batch[:,-1:].cpu())#[N]
        elif batch.shape[1]==2:
            target_data=np.append(target_data, batch[:,:].cpu())#[N,2]
    model = KMeans(n_clusters,random_state=42)
    if len(target_data.shape)==1:
        model.fit(target_data[:,np.newaxis])
    else:
        model.fit(target_data)
    return model.cluster_centers_# [n_clusters,1] or [n_clusters,2]
