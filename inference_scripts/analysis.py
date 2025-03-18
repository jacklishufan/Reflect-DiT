import glob
import pathlib
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def read_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)
    
import sys
# root = sys.argv[1]

# gptprompts = pd.read_json(gpt,lines=True)
# prompts = gptprompts.prompt.unique()
from prettytable import PrettyTable

class Cache:
    include_prompts = None
def get_plot(root,include_prompts=None): 
    # pass include_prompts for DPG result to obtain subset numbers
    if '.csv' in root:
        df = pd.read_csv(root)
        return df.index+1,df['pass_rate']
    elif 'jsonl' in root:
        df = pd.read_json(root,lines=True)
        if 'gdino' in root:
            df['correct_verifier'] = df['gdino_correct'] == 1
        elif 'gpt' in root:
            df['correct_verifier'] = df['gpt_feedback'].str.contains('incorrect')
        data = df.to_dict(orient='records')
        new_data = []
        passed = set()
        for row in data:
            if row['prompt'] in passed:
                continue
            if row['correct_verifier']:
                new_data.append(row)
                passed.add(row['prompt'])
            else:
                new_data.append(row)
        df = pd.DataFrame(new_data)
    else:
        files = glob.glob(os.path.join(root,'**/*.feedback.json'),recursive=True)
        # breakpoint()
        results = []
        for f in files:
            load_results = read_json(f)
            if type(load_results) == list:
                        feedback_results = load_results[-1]
            else:
                feedback_results = load_results
            if 'filename' not in feedback_results:
                feedback_results['filename'] = f.replace('.feedback.json','')
                
            if 'prompt' not in feedback_results:
                raise ValueError()
            results.append(feedback_results)
            
        df = pd.DataFrame(results)
    grouped_df = df.groupby('prompt').agg(
        count=('correct', 'size'),
        accuracy=('correct', 'max')
    ).reset_index()
    max_len = grouped_df['count'].max()
    
    df['gen_index'] = [int(os.path.basename(x).replace('.png','')) for x in df.filename]
    unique_prompts = df.prompt.unique()
    unique_prompts = [x for x in unique_prompts if type(x) == str]
    extra_rows = []
    for p in unique_prompts:
        last_res = df[df.prompt == p].sort_values('gen_index').to_dict(orient='records')
        last_res = last_res[-1]
        last_idx = last_res['gen_index']
        for gen_idx in range(last_idx+1,max_len):
            copy_dict = dict(**last_res)
            copy_dict['gen_index'] = gen_idx
            extra_rows.append(copy_dict)
        pass
    total_prompts = len(unique_prompts)
    pass_at_k = []
    xx = []
    extra_df = pd.DataFrame(extra_rows)
    df = pd.concat([df,extra_df])
    filter_zero = include_prompts is not None
    out_root = 'outputs'
    df.to_csv(os.path.join(out_root,os.path.basename(root)+'.csv'))
    if filter_zero:
        print("Remaining:",len(include_prompts))
    for i in range(1,max_len+1):
        partial_df = df[df['gen_index']==(i-1)]
        assert len(partial_df) == total_prompts,(len(partial_df),total_prompts)
        if filter_zero:
            partial_df = partial_df[partial_df.prompt.isin(include_prompts)]
        pass_at_k_local = partial_df['correct'].mean()
        pass_at_k.append(pass_at_k_local)
        xx.append(i)
    print(root)
    final_res = df[df['gen_index']==(max_len-1)]
    if 'tag' in final_res.columns:
        by_category = final_res.groupby('tag').agg(
            acc=('correct','mean')
        )
        res = by_category.to_dict()['acc']
        res['overall'] = final_res['correct'].mean()

        table = PrettyTable()
        table.add_row(list(res.values()))
        table.field_names = list(res.keys())
        
        print(table)
        table = PrettyTable()
        #xx = np.array(xx) + 1 # 0 index to 1 index
        table.field_names = [f'N={x}' for x in xx[3::4]]
        table.add_row(pass_at_k[3::4])
        print(table)
        return (xx[1::2],pass_at_k[1::2])
    return (xx,pass_at_k)
plt.figure(figsize=(8, 6),dpi=200)
def remove_common_prefix(str_list):
    if not str_list:
        return str_list
    prefix = os.path.commonprefix(str_list)
    return [s[len(prefix):] for s in str_list]
DPG_HARD_56 = pd.read_csv('data/dpg/subsets/dpg-hard-56.csv')
DPG_HARD_246 = pd.read_csv('data/dpg/subsets/dpg-hard-246.csv')
# compute results only on subset if necessary
if '--hard56' in sys.argv:
    include_prompts = DPG_HARD_56.prompt
elif '--hard246' in sys.argv:
    include_prompts = DPG_HARD_246.prompt
else:
    include_prompts = None
    
plt.plot(*get_plot(sys.argv[1],include_prompts=include_prompts), marker='o', linestyle='-')
plt.xticks(range(2,21,2))
plt.legend(['Result'],fontsize=15)
plt.xlabel('Number of Samples',fontsize=20)
plt.ylabel('GenEval Score',fontsize=20)
plt.title('Comparison with Finetuning Methods',fontsize=20)
plt.grid(True)
plt.savefig('outputs/result.jpg')
