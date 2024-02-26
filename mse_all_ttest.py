import numpy as np
from scipy import stats
import pdb
import os
import argparse
import pickle

# best: 平均誤差が一番低い方法の全テスト誤差
# ref: 比較対象の方法の全テスト誤差
# thread: 有意水準、デフォルト：0.025（95%）
dir_file=os.getcwd()
dir=os.getcwd()
if not os.path.exists(f'{dir}/log/SE'):# 無ければ
    os.makedirs(f'{dir}/log/SE') 
log_file=f'{dir}/log/SE/All_SE_log.txt'
methods=["const","exp","pc","omi","THP","proposed","mv6","anhp"]
def open_file(file_path):
    with open(f'{file_path}', 'rb') as pfile:
        file_value = pickle.load(pfile)
    return file_value

def get_GT(gene):
    return open_file(f"{dir_file}/pickled/GT/{gene}/GT")

def get_const_ABS(gene):
    model_name="const"
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_ABS_Error")
def get_const_ll(gene):
    model_name="const"
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{gene}_ll")
def get_const_nonll(gene):
    model_name="const"
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{gene}_nonll")
def get_const_pred(gene):
    model_name="const"
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")

def get_exp_ABS(gene):
    model_name="exp"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_ABS_Error")
def get_exp_ll(gene):
    model_name="exp"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{gene}_ll")
def get_exp_nonll(gene):
    model_name="exp"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{gene}_nonll")
def get_exp_pred(gene):
    model_name="exp"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")
    
def get_pc_ABS(gene):
    model_name="pc"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_ABS_Error")
def get_pc_ll(gene):
    model_name="pc"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{gene}_ll")
def get_pc_nonll(gene):
    model_name="pc"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{gene}_nonll")
def get_pc_pred(gene):
    model_name="pc"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")

def get_omi_ll(gene):
    method_name="omi"
    return open_file(f"{dir_file}/pickled/FT_PP/{method_name}/{gene}/{gene}_ll")
def get_omi_nonll(gene):
    method_name="omi"
    return open_file(f"{dir_file}/pickled/FT_PP/{method_name}/{gene}/{gene}_nonll")
def get_omi_ABS(gene):
    model_name="omi" 
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_ABS_Error")
def get_omi_pred(gene):
    model_name="omi"    
    return open_file(f"{dir_file}/pickled/FT_PP/{model_name}/{gene}/{model_name}_{gene}_pred")

def get_THP_ABS(gene):
    imp="h173"
    method_name="THP"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_ABS")
def get_THP_ll(gene):
    imp="h173"
    method_name="THP"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_ll")
def get_THP_nonll(gene):
    imp="h173"
    method_name="THP"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_nonll")
def get_THP_pred(gene):
    imp="h173"
    method_name="THP"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_pred")

def get_mv3_ABS(gene, imp, what):
    imp="h123"
    method_name="mv3"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_ABS")
def get_mv3_ll(gene):
    imp="h123"
    method_name="mv3"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_ll")
def get_mv3_nonll(gene):
    imp="h123"
    method_name="mv3"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_nonll")
def get_mv3_pred(gene):
    imp="h173"
    method_name="mv3"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_pred")
 
def get_mv6_ABS(gene, imp, what):
    imp="h173"
    method_name="mv6"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_ABS")
def get_mv6_ll(gene):
    imp="h173"
    method_name="mv6"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_ll")
def get_mv6_nonll(gene):
    imp="h173"
    method_name="mv6"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_nonll")
def get_mv6_pred(gene):
    imp="h173"
    method_name="mv6"
    return open_file(f"{dir_file}/pickled/THP/{gene}/THP_{method_name}{imp}{gene}_pred")

def get_anhp_any(gene,any_name):
    if gene=="911_1_Address":
        return open_file(f"/data1/nishizawa/anhp-andtt/domains/call1/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-12883/results_test{any_name}")
    elif gene=="911_2_Address":
        return open_file(f"/data1/nishizawa/anhp-andtt/domains/call2/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-25483/results_test{any_name}")
    elif gene=="911_3_Address":
        return open_file(f"/data1/nishizawa/anhp-andtt/domains/call3/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-6503/results_test{any_name}")
    elif gene=="h1":
        return open_file(f"/data1/nishizawa/anhp-andtt/domains/h1/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-5144/results_test{any_name}")
    elif gene=="h_fix05":
        return open_file(f"/data1/nishizawa/anhp-andtt/domains/h_fix05/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-11139/results_test{any_name}")
    elif gene=="jisin":
        return open_file(f"/data1/nishizawa/anhp-andtt/domains/jisin/ContKVLogs/h-1_me-100_d_model-64_dp-0.1_teDim-64_layer-4_lr-0.0001_seed-32_ignoreFirst-False_id-31826/results_test{any_name}")

def get_proposed_ABS(gene):
    return open_file(f"{dir_file}/pickled/proposed/{gene}/_{gene}_ABS_Error")
def get_proposed_ll(gene):
    return open_file(f"{dir_file}/pickled/proposed/{gene}/_{gene}ll")
def get_proposed_nonll(gene):
    return open_file(f"{dir_file}/pickled/proposed/{gene}/_{gene}nonll")
def get_proposed_pred(gene):
    return open_file(f"{dir_file}/pickled/proposed/{gene}/_{gene}_pred")

def write_log(file_path, string):
    with open(file_path, 'a') as f:
        f.write(f'{string}\n')
    
def paired_ttest(best, refs, min_method_num, thread=0.025,gene=""):
    print(f'significance level:{thread}')
    with open(log_file, 'a') as f:
        f.write(f'{gene}\nsignificance level:{thread}\n')
    for i in range(len(refs)):
        if i == min_method_num:
            continue
        res = stats.ttest_rel(best, refs[i])
        print(f'method {[i]} {methods[i]}: pvalue:{np.round(res.pvalue,5)}')
        with open(log_file, 'a') as f:
            f.write(f'method {methods[i]}: pvalue:{np.round(res.pvalue,5)}')
        if res.pvalue < thread:
            print(f" worse!")
            with open(log_file, 'a') as f:
                f.write(f" worse!\n")
        else:
            print(f"equivalent!")
            with open(log_file, 'a') as f:
                f.write(f" equivalent!\n")
# 有意水準: 95%


def main():
    with open(log_file, 'w') as f:
        f.write(f'')
    genes=["jisin","h1","h_fix05","call"]
    for gene_select in genes:
        if gene_select=="call":
            gene_list=["911_1_Address","911_2_Address","911_3_Address"]
        else:
            gene_list=[gene_select]
        #pdb.set_trace()
        errors=None
        errors_list=None
        eve_list=None
        for gene in gene_list:
            print(gene)
            GT=get_GT(gene)
            anhp_GT=np.array(get_anhp_any(gene,"GT_his"))
            what=np.array(get_anhp_any(gene,""))
            # what.item().keys()
            # dict_keys(['loglik', 'loglik_token', 'type_ll_token', 'time_ll_token', 'pred'])
            # what.item()[]"pred"]
            const_pred=get_const_pred(gene)
            exp_pred=get_exp_pred(gene)
            pc_pred=get_pc_pred(gene)
            omi_pred=get_omi_pred(gene)
            proposed_pred=get_proposed_pred(gene)
            THP_pred=get_THP_pred(gene)
            #mv3_pred=get_mv3_pred(gene)
            mv6_pred=get_mv6_pred(gene)
            anhp_pred=np.array(get_anhp_any(gene,"pred_his"))
            #
            
            const_ABS=get_const_ABS(gene)
            exp_ABS=get_exp_ABS(gene)
            pc_ABS=get_pc_ABS(gene)
            omi_ABS=get_omi_ABS(gene)
            proposed_ABS=get_proposed_ABS(gene)
            anhp_ABS=abs(GT-anhp_pred)
            
            const_SE=((const_pred-GT)**2).reshape(1,-1)#np.sqrt(const_SE.mean())
            exp_SE=((exp_pred-GT)**2).reshape(1,-1)#np.sqrt(exp_SE.mean())
            pc_SE=((pc_pred-GT)**2).reshape(1,-1)#np.sqrt(pc_SE.mean())
            omi_SE=((omi_pred-GT)**2).reshape(1,-1)#np.sqrt(omi_SE.mean())
            THP_SE=((THP_pred-GT)**2).reshape(1,-1)#np.sqrt(THP_SE.mean())
            proposed_SE=((proposed_pred-GT)**2).reshape(1,-1)#np.sqrt(proposed_SE.mean())
            #mv3_SE=((mv3_pred-GT)**2).reshape(1,-1)#np.sqrt(mv3_SE.mean())
            mv6_SE=((mv6_pred-GT)**2).reshape(1,-1)#np.sqrt(mv6_SE.mean())
            anhp_SE=(anhp_ABS**2).reshape(1,-1)#np.sqrt(anhp_SE.mean())
            #pdb.set_trace()
            const_ll=get_const_ll(gene)
            exp_ll=get_exp_ll(gene)
            pc_ll=get_pc_ll(gene)
            omi_ll=get_omi_ll(gene)
            proposed_ll=get_proposed_ll(gene)
            THP_ll=get_THP_ll(gene)
            #mv3_ll=get_mv3_ll(gene)
            mv6_ll=get_mv6_ll(gene)
            
            const_nonll=get_const_nonll(gene)
            exp_nonll=get_exp_nonll(gene)
            pc_nonll=get_pc_nonll(gene)
            omi_nonll=get_omi_nonll(gene)
            proposed_nonll=get_proposed_nonll(gene)
            THP_nonll=get_THP_nonll(gene)
            #mv3_nonll=get_mv3_nonll(gene)
            mv6_nonll=get_mv6_nonll(gene)
            
            n_data=const_ll.shape[0]
            llplus=1e-6
            nonplus=1e-6*GT
            #pdb.set_trace()
            #np.log(np.exp(THP_eve*-1)+1e-8).mean()*-1
            
            #ver1
            const_ll=np.log(np.exp(const_ll)+llplus)
            const_nonll=np.log(np.exp(const_nonll)+nonplus)
            exp_ll=np.log(np.exp(exp_ll)+llplus)
            exp_nonll=np.log(np.exp(exp_nonll)+nonplus)
            pc_ll=np.log(np.exp(pc_ll)+llplus)
            pc_nonll=np.log(np.exp(pc_nonll)+nonplus)
            omi_ll=np.log(np.exp(omi_ll)+llplus)
            omi_nonll=np.log(np.exp(omi_nonll)+nonplus)
            THP_ll=np.log(np.exp(THP_ll)+llplus)
            THP_nonll=np.log(np.exp(THP_nonll)+nonplus)
            proposed_ll=np.log(np.exp(proposed_ll)+llplus)
            proposed_nonll=np.log(np.exp(proposed_nonll)+nonplus)
            #mv3_ll=np.log(np.exp(mv3_ll)+llplus)
            #mv3_nonll=np.log(np.exp(mv3_nonll)+nonplus)
            mv6_ll=np.log(np.exp(mv6_ll)+llplus)
            mv6_nonll=np.log(np.exp(mv6_nonll)+nonplus)
            
            const_eve=-(const_ll-const_nonll).reshape(1,-1)
            exp_eve=-(exp_ll-exp_nonll).reshape(1,-1)
            pc_eve=-(pc_ll-pc_nonll).reshape(1,-1)
            omi_eve=-(omi_ll-omi_nonll).reshape(1,-1)
            THP_eve=-(THP_ll-THP_nonll).reshape(1,-1)
            proposed_eve=-(proposed_ll-proposed_nonll).reshape(1,-1)
            #mv3_eve=-(mv3_ll-mv3_nonll).reshape(1,-1)
            mv6_eve=-(mv6_ll-mv6_nonll).reshape(1,-1)
            anhp_eve=-((np.array(what.item()["loglik"])[:,0])).reshape(1,-1)
            
            tmp_errors=np.concatenate([const_SE,exp_SE,pc_SE,omi_SE,THP_SE,proposed_SE,mv6_SE,anhp_SE],axis=0)
            tmp_eves=np.concatenate([const_eve,exp_eve,pc_eve,omi_eve,THP_eve,proposed_eve,mv6_eve,anhp_eve],axis=0)

            if errors_list is not None:
                errors_list=np.concatenate([errors_list,tmp_errors],axis=1)
            else:
                errors_list=tmp_errors
            if eve_list is not None:
                eve_list=np.concatenate([eve_list,tmp_eves],axis=1)
            else:
                eve_list=tmp_eves
        print(f"const:{np.round(np.sqrt((errors_list[0]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[0])),decimals=3)})"
               f", ll:{np.round(eve_list[0].mean(),decimals=3)}({np.round(np.std(eve_list[0]),decimals=3)})")

        print(f"exp:{np.round(np.sqrt((errors_list[1]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[1])),decimals=3)})"
               f", ll:{np.round(eve_list[1].mean(),decimals=3)}({np.round(np.std(eve_list[1]),decimals=3)})")

        print(f"pc:{np.round(np.sqrt((errors_list[2]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[2])),decimals=3)})"
               f", ll:{np.round(eve_list[2].mean(),decimals=3)}({np.round(np.std(eve_list[2]),decimals=3)})")
       
        print(f"omi:{np.round(np.sqrt((errors_list[3]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[3])),decimals=3)})"
               f", ll:{np.round(eve_list[3].mean(),decimals=3)}({np.round(np.std(eve_list[3]),decimals=3)})")
       
        print(f"THP:{np.round(np.sqrt((errors_list[4]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[4])),decimals=3)})"
               f", ll:{np.round(eve_list[4].mean(),decimals=3)}({np.round(np.std(eve_list[4]),decimals=3)})")
       
        print(f"prop:{np.round(np.sqrt((errors_list[5]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[5])),decimals=3)})"
               f", ll:{np.round(eve_list[5].mean(),decimals=3)}({np.round(np.std(eve_list[5]),decimals=3)})")
       
        # print(f"mv3:{np.round(np.sqrt((errors_list[6]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[6])),decimals=3)})"
        #        f", ll:{np.round(eve_list[6].mean(),decimals=3)}({np.round(np.std(eve_list[6]),decimals=3)})")
       
        print(f"mv6:{np.round(np.sqrt((errors_list[6]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[6])),decimals=3)})"
               f", ll:{np.round(eve_list[6].mean(),decimals=3)}({np.round(np.std(eve_list[6]),decimals=3)})")
        
        print(f"anhp:{np.round(np.sqrt((errors_list[7]).mean()),decimals=3)}({np.round(np.sqrt(np.std(errors_list[7])),decimals=3)})"
               f", ll:{np.round(eve_list[7].mean(),decimals=3)}({np.round(np.std(eve_list[7]),decimals=3)})")
        
        
        min_ind = np.argmin(np.sqrt(errors_list.mean(axis=1)))
        exclude_ind_mask=np.ones(errors_list.shape[0],dtype=bool)
        exclude_ind_mask[min_ind]=False
        
        paired_ttest(errors_list[min_ind], errors_list, min_ind, thread=0.025,gene=f"{gene_select}_mse")
        
        min_ind = np.argmin((eve_list.mean(axis=1)))
        exclude_ind_mask=np.ones(eve_list.shape[0],dtype=bool)
        exclude_ind_mask[min_ind]=False
        paired_ttest(eve_list[min_ind], eve_list, min_ind, thread=0.025,gene=f"{gene_select}_ll")


if __name__ == "__main__":
    main()