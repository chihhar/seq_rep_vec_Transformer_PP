import os
import pdb
import math
import numpy as np
import torch
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt

dir_folder=os.getcwd()

def open_file(file_path):
    with open(f'{file_path}', 'rb') as pfile:
        file_value = pickle.load(pfile)
    return file_value

def get_GT(gene):
    return open_file(f"{dir_folder}/pickled/GT/{gene}/GT")

def get_const_pred(gene):
    model_name="const"
    return open_file(f"{dir_folder}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")

def get_exp_pred(gene):
    model_name="exp"    
    return open_file(f"{dir_folder}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")

def get_pc_pred(gene):
    model_name="pc"    
    return open_file(f"{dir_folder}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")

def get_omi_pred(gene):
    model_name="omi"    
    return open_file(f"{dir_folder}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")

def get_THP_pred(gene):
    imp="h173"
    method_name="THP"
    return open_file(f"{dir_folder}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_pred")

def get_mv3_pred(gene):
    imp="h123"
    method_name="mv3"
    return open_file(f"{dir_folder}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_pred")

def get_mv6_pred(gene):
    imp="h173"
    method_name="mv6"
    return open_file(f"{dir_folder}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_pred")

def get_proposed_pred(gene):
    if gene=="911_1_Address":
        return open_file(f"{dir_folder}/pickled/proposed/{gene}/_911_1_Address_pred")
    elif gene=="911_2_Address":
        return open_file(f"{dir_folder}/pickled/proposed/{gene}/_911_2_Address_pred")
    elif gene=="911_3_Address":
        return open_file(f"{dir_folder}/pickled/proposed/{gene}/_911_3_Address_pred")
    elif gene=="h1":
        return open_file(f"{dir_folder}/pickled/proposed/{gene}/_h1_pred")
    elif gene=="h_fix05":
        return open_file(f"{dir_folder}/pickled/proposed/{gene}/_h_fix05_pred")
    elif gene=="jisin":
        return open_file(f"{dir_folder}/pickled/proposed/{gene}/_jisin_pred")
def get_anhp_any(gene,any_name):
    if gene=="911_1_Address":
        return open_file(f"anhp-andtt/domains/call1/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-12883/results_test{any_name}")
    elif gene=="911_2_Address":
        return open_file(f"anhp-andtt/domains/call2/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-25483/results_test{any_name}")
    elif gene=="911_3_Address":
        return open_file(f"anhp-andtt/domains/call3/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-6503/results_test{any_name}")
    elif gene=="h1":
        return open_file(f"anhp-andtt/domains/h1/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-5144/results_test{any_name}")
    elif gene=="h_fix05":
        return open_file(f"anhp-andtt/domains/h_fix05/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-11139/results_test{any_name}")
    elif gene=="jisin":
        return open_file(f"anhp-andtt/domains/jisin/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-31826/results_test{any_name}")


def event_GT_plot(gene):
    # fig.4
    GT=get_GT(gene)
    const_pred=get_const_pred(gene)
    exp_pred=get_exp_pred(gene)
    pc_pred=get_exp_pred(gene)
    omi_pred=get_omi_pred(gene)
    THP_pred=get_THP_pred(gene)
    #THP3_pred=get_mv3_pred(gene)
    THP6_pred=get_mv6_pred(gene)
    pro_pred=get_proposed_pred(gene)
    anhp_pred=np.array(get_anhp_any(gene,"pred_his"))
    # pdb.set_trace()
    plt.figure(figsize=(8.4,1.68),dpi=300)
    
    plt.plot(range(GT[0:100].shape[0]),GT[0:100],label="ground-truth",      color="k",lw=1.5,linestyle="solid",marker="*", markersize=2)
    plt.plot(range(omi_pred[0:100].shape[0]),omi_pred[0:100],label="FTPP",color="g",lw=1.0,linestyle="dotted",marker="^", markersize=2)
    plt.plot(range(THP_pred[0:100].shape[0]),THP_pred[0:100],label="THP",color="b",lw=1.0,linestyle="dashdot",marker="x", markersize=2)
    plt.plot(range(anhp_pred[0:100].shape[0]),anhp_pred[0:100],label=f"A-NHP",color="c",lw=1.0,linestyle="dashdot",marker="v", markersize=2)
    plt.plot(range(pro_pred[0:100].shape[0]),pro_pred[0:100],label=f"proposed",color="r",lw=1.0,linestyle="dashed",marker="o", markersize=2)
    plt.xlabel(r" event ID",fontsize=12,labelpad=0,loc="center")
    plt.ylabel(r"elapsed time",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if gene=="h1":
        plt.legend(fontsize=10, loc='upper right')
    if not os.path.exists(f"plot/all_eve_pred/{gene}/"):
        os.makedirs(f"plot/all_eve_pred/{gene}/")

    plt.savefig(f"{dir_folder}/plot/all_eve_pred/{gene}/{gene}_ID_time_plot_matome.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir_folder}/plot/all_eve_pred/{gene}/{gene}_ID_time_plot_matome.svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir_folder}/plot/all_eve_pred/{gene}/{gene}_ID_time_plot_matome.png", bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def compute_event(event):
    """ Log-likelihood of events. """
    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    result = np.log(event)
    return result


def naiseki_plot():
    # fig.5
    gene="h_fix05"
    file_path=f"{dir_folder}/pickled/THP/{gene}"
   
    lS=open_file(f"{file_path}/THP__h_fix05_naiseki_lS")
    lH=open_file(f"{file_path}/THP__h_fix05_naiseki_lH")
    
    initial_S=open_file(f"{file_path}/THP__h_fix05_naiseki_initial_S")
        
    H0=open_file(f"{file_path}/THP__h_fix05_naiseki_H0")
    mk=["^","x","o","*"]
    c_name=["m","b","g","r"]
    lww=[1.0,1.0,1.0,1.5]
    ev_num=H0.size(0)
    temp_sim=torch.cosine_similarity(initial_S,H0)

    plt.figure(figsize=(8,5))
    plt.ylim(-1.01,1.05)
    plt.plot(range(2,ev_num+2),temp_sim[:ev_num].cpu().detach(),lw=1.0,label=r"initial",color="k",marker="D", markersize=7)
    for loop_layer in range(lS.shape[0]):
        temp_sim=torch.cosine_similarity(lS[loop_layer],lH[loop_layer])
        plt.plot(range(2,ev_num+2),temp_sim[:ev_num].cpu().detach(),label=f"layer{loop_layer+1}",lw=lww[loop_layer],marker=mk[loop_layer], markersize=7,color=c_name[loop_layer])
    plt.xlabel(f"past event index",fontsize=18)
    plt.ylabel(f"similarity",fontsize=18)
    plt.legend(fontsize=12, loc='lower right')
    plt.xticks([2,10,20,30])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18, loc='lower right')
    if not os.path.exists(f"plot/ronb/"):
        os.makedirs(f"plot/ronb/")
    plt.savefig("plot/ronb/THP_Event_normDot_histi_.png", bbox_inches='tight', pad_inches=0)
    plt.savefig("plot/ronb/THP_Event_normDot_histi_.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig("plot/ronb/THP_Event_normDot_histi_.svg", bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    
    file_path=f"{dir_folder}/pickled/proposed/{gene}"
    with open(f"{file_path}/proposed__h_fix05_naiseki_H0", 'rb') as file:
            H0=pickle.load(file)
    with open(f"{file_path}/proposed__h_fix05_naiseki_initial_S", 'rb') as file:
        initial_S=pickle.load(file)
    with open(f"{file_path}/proposed__h_fix05_naiseki_lS", 'rb') as file:
        lS=pickle.load(file)
    with open(f"{file_path}/proposed__h_fix05_naiseki_lH", 'rb') as file:
        lH=pickle.load(file)
            
    
    for r_num in range(initial_S.shape[0]):
        
        plt.figure(figsize=(8,5))
        plt.xticks([2,10,20,30])
        plt.ylim(-1.01,1.05)
        temp_sim=torch.cosine_similarity(initial_S[r_num],H0)
        plt.plot(range(2,ev_num+2),temp_sim[:ev_num].cpu().detach(),lw=1.0,label=r"initial",color="k",marker="D", markersize=7)
        for ln in range(lS.shape[0]):
            if ln==0:
                mk="^"
                c_name="m"
            elif ln==1:
                mk="x"
                c_name="b"
            elif ln==2:
                mk="o"
                c_name="g"
            elif ln==3:
                mk="*"
                c_name="r"
            temp_sim=torch.cosine_similarity(lS[ln][r_num],lH[ln])
            if ln==3:
                lww=1.5
            else:
                lww=1.0
            plt.plot(range(2,ev_num+2),temp_sim[:ev_num].cpu().detach(),label=f"layer{ln+1}",lw=lww,marker=mk, markersize=7,color=c_name)
        plt.xlabel(r"past event index",fontsize=18)
        plt.ylabel("similarity",fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.savefig(f"plot/ronb/S{r_num+1}naiseki_histID.svg", bbox_inches='tight', pad_inches=0)
        plt.savefig(f"plot/ronb/S{r_num+1}naiseki_histID.pdf", bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
       
def anc_naiseki_plot():
    #fig.6
    with open(f"pickled/proposed/h_fix05/anc/h_fix05proposed_initial.pkl", "rb") as file:
        initial_A=pickle.load(file)
    with open(f"pickled/proposed/h_fix05/anc/h_fix05proposed_S1.pkl", 'rb') as file:
        S1=pickle.load(file)
    with open(f"pickled/proposed/h_fix05/anc/h_fix05proposed_output_A","rb") as file:
        output_A=pickle.load(file)
    
    x_val=np.array([1,2,3])
    color_len=["m","b","g","r"]
    patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    
    for loop_anc_num in range(initial_A.shape[0]):
        plt.figure(figsize=(8,5))
        plt.ylim(0,1.0)
        plt.xlabel(f"seq-rep vector index",fontsize=18)
        plt.ylabel(f"similarity",fontsize=18)
        temp_sim=torch.cosine_similarity(S1,initial_A[loop_anc_num,:])
        plt.bar(x_val-0.2,torch.softmax(temp_sim,dim=0).cpu().detach(),width=0.1,label=f"initial",color="k")
        for loop_layer_num in range(output_A.shape[0]):
            temp_sim=torch.cosine_similarity(S1,output_A[loop_layer_num,loop_anc_num,:])
            plt.bar(x_val-0.1+loop_layer_num*0.1,torch.softmax(temp_sim,dim=0).cpu().detach(),width=0.1,label=f"layer{loop_layer_num+1}",color=color_len[loop_layer_num],hatch=patterns[loop_layer_num])
        plt.xticks(x_val,["1","2","3"],fontsize=18)
        plt.yticks(fontsize=18)
        if loop_anc_num==0:
            plt.legend(fontsize=18, loc='upper right')
        plt.savefig(f"plot/ronb/poiA{loop_anc_num+1}naiseki_histID.pdf", bbox_inches='tight', pad_inches=0)
        plt.savefig(f"plot/ronb/poiA{loop_anc_num+1}naiseki_histID.svg", bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
def plot_TSNE():
    # fig.7
    gene="h_fix05"
    imp="_"
    for where in ["rep", "anc"]:
        plt.figure(figsize=(4,4),dpi=100)
        plot_file = f"{dir_folder}/plot/t_SNE/{where}"
        with open(f"{dir_folder}/pickled/proposed/{gene}/_{gene}_True","rb") as file:
            GT_his=pickle.load(file)
        with open(f"{dir_folder}/pickled/proposed/{gene}/{where}/proposed_{imp}{gene}_use_umap_2Dvector.pkl","rb") as file:
            db=pickle.load(file)
        plt.scatter(db[:,0], db[:,1], c=GT_his,cmap='gist_stern')
        plt.savefig(f"{plot_file}/GT_trainUMAP{gene}{imp}_c{where}.pdf", bbox_inches='tight', pad_inches=0)
        plt.savefig(f"{plot_file}/GT_trainUMAP{gene}{imp}_c{where}.svg", bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()
    ### THP #########
    #THP__h_fix05_use_umap_2Dvector
    with open(f"{dir_folder}/pickled/THP/{gene}/THP_{imp}{gene}_use_umap_2Dvector.pkl", 'rb') as file:
        THP=pickle.load(file)
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(THP[:,0], THP[:,1], c=GT_his,cmap='gist_stern')
    plt.colorbar()
    plt.savefig(f"{dir_folder}/plot/t_SNE/GT_THP_trainUMAP{gene}{imp}.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir_folder}/plot/t_SNE/GT_THP_trainUMAP{gene}{imp}.svg", bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def inten_plot():
    # fig.8
    with open(f"{dir_folder}/pickled/proposed/h_fix05/proposed__h_fix05_intensity", "rb") as file:
        log_likelihood_history=pickle.load(file)
    with open(f"{dir_folder}/pickled/proposed/h_fix05/target_intensity", "rb") as file:
        target_history=pickle.load(file)
    with open(f"{dir_folder}/pickled/proposed/h_fix05/True_intensity", "rb") as file:
        calc_log_l_history=pickle.load(file)
    with open(f"{dir_folder}/pickled/proposed/h_fix05/eventtime_intensity", "rb") as file:
        cumsum_tau=pickle.load(file)
    with open(f"{dir_folder}/pickled/anhp/calc_intensity", 'rb') as file:
        anhp_ll=pickle.load(file)
    #anhp_ll=np.log(anhp_ll)
    # THP load
    THP_ll=np.load(f"{dir_folder}/pickled/THP/h_fix05/THPh164pre_l4h1_calc_intensity.npy")
    anhp_ll=compute_event(anhp_ll)
    # log_likelihood_history=np.exp(log_likelihood_history)
    # THP_ll=np.exp(THP_ll)
    # anhp_ll=np.exp(anhp_ll)
    # calc_log_l_history=np.exp(calc_log_l_history)
    
    plt.figure(figsize=(6,6),dpi=300)
    plt.figure(figsize=(10,10))
    
    target_history=np.array([x.cpu().numpy() for x in target_history])
    plt.plot(target_history,calc_log_l_history,label=r"ground-truth",color="k",lw=2.25,alpha=0.8)
    plt.scatter(cumsum_tau.cpu(),torch.zeros(cumsum_tau.shape)-1.9,marker='x',color="k",label="event-time", s=80)
    
    #THP
    plt.plot(target_history,THP_ll,label=r"THP",color="b",linestyle="dotted",lw=1.0)
    plt.plot(target_history,anhp_ll,label=r"A-NHP",color="c",linestyle="solid",lw=1.0)
    #proposed hazard
    plt.plot(target_history,log_likelihood_history,label=r"proposed method",color="r",linestyle="dashed",lw=1.0)
    plt.ylim(-2.0,2.0)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r"time", fontsize=18)
    plt.ylabel(r"log-intensity", fontsize=18)
    plt.legend(fontsize=18)
    if not os.path.exists(f"plot/all_eve_pred/"):
        os.makedirs(f"plot/all_eve_pred/")
    plt.savefig(f"{dir_folder}/plot/all_eve_pred/toydata_intensity.pdf",bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir_folder}/plot/all_eve_pred/toydata_intensity.svg",bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{dir_folder}/plot/all_eve_pred/toydata_intensity.png",bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

def main():
    naiseki_plot()# figure 5.
    anc_naiseki_plot()# figure 6.
    plot_TSNE()# figure 7.
    inten_plot()# figure 8.
    
    #figure 4.
    gene_list=["h1","h_fix05","jisin","call"]
    for gene_select in gene_list:
        if gene_select=="call":
            gene_list=["911_1_Address","911_2_Address","911_3_Address"]
        else:
            gene_list=[gene_select]
        for gene in gene_list:
            # event and GT
            event_GT_plot(gene)
    

if __name__=="__main__":
    plt.switch_backend('agg')
    plt.figure()
    main()
    main()