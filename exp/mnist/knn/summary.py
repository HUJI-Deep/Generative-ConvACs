import pandas as pd
import os
import re


if __name__ == '__main__':
    all_res_files = [fname for fname in os.listdir(".") if fname.startswith('out')]
    all_dfs = pd.DataFrame()
    for res_file in all_res_files:
        try:
            df = pd.read_csv(res_file,sep=' ',header=None)
            all_dfs = all_dfs.append(df)
        except:
            print "skipped ", res_file
    
    
    rects_re = re.compile('.*image_data_minrects_(\d)_maxrects_(\d)_minwidth_(\d+)_maxwidth_(\d+)')
    iid_re = re.compile('.*image_data_mar_(.*)')

    csv_df = pd.DataFrame()
    for index, row in all_dfs.iterrows():
        rect_match = rects_re.match(row[0])
        if rect_match:
            min_n,max_n,min_d,max_d = rect_match.groups()
            new_res = pd.DataFrame({'num_rects':[ min_n], 'rect_size': [min_d],
                                    'accuracy': [row[2]], 'iid_prob':[ -1 ]})
            csv_df = csv_df.append(new_res)

    for index, row in all_dfs.iterrows():
        iid_match = iid_re.match(row[0])
        if iid_match:
            iid_prob = iid_match.groups()[0]
            new_res = pd.DataFrame({'iid_prob':[ iid_prob], 'accuracy': [row[2]]
            ,'num_rects':[ -1 ], 'rect_size': [ -1 ]})
            csv_df = csv_df.append(new_res)
            
    csv_df.to_csv('results.csv')
       
    
    