import pandas as pd
import os
import re

FLOAT_RE = 'nan|[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
acc_re = re.compile(r'accuracy = '+r'(^'+FLOAT_RE+'$)')

def extract_accuracy(fname):
    acc = -1
    with open(fname, 'rb') as f:
        for line in f.readlines():
            res = acc_re.findall(line)
            if res:
                acc = float(res[0][0])
                return acc
                
if __name__ == '__main__':
    all_res_files = [fname for fname in os.listdir(".") if fname.endswith('out')]
    all_res = []
    for res_file in all_res_files:
        try:
            acc = extract_accuracy(res_file)
            all_res.append(pd.Series( {'fname': os.path.splitext(res_file)[0], 'acc': float(acc)} ))     
        except:
            print "skipped ", res_file
    
    
    rects_re = re.compile('.*image_data_minrects_(\d)_maxrects_(\d)_minwidth_(\d+)_maxwidth_(\d+)')
    iid_re = re.compile('.*image_data_mar_(.*)')
    
    
    res_df = pd.DataFrame(all_res)

    csv_df = pd.DataFrame()
    for index, row in res_df.iterrows():
        rect_match = rects_re.match(row['fname'])
        iid_match = iid_re.match(row['fname'])
        if rect_match:
            min_n,max_n,min_d,max_d = rect_match.groups()
            new_res = pd.DataFrame({'num_rects':[ int(min_n)], 'rect_size': [int(min_d)],
                                    'accuracy': [row['acc']], 'iid_prob':[ -1 ]})
            csv_df = csv_df.append(new_res)
        if iid_match:
            iid_prob = iid_match.groups()[0]
            new_res = pd.DataFrame({'iid_prob':[ float(iid_prob)], 'accuracy': [row['acc']]
            ,'num_rects':[ -1 ], 'rect_size': [ -1 ]})
            csv_df = csv_df.append(new_res)
                        
    csv_df.to_csv('results.csv')
    csv_df.sort_values(by=['iid_prob'], inplace=True)
    csv_df.sort_values(by=['num_rects','rect_size'], inplace=True)
    
    iid_df = csv_df[csv_df['iid_prob'] != -1]
    iid_df.sort_values(by=['iid_prob'])
    iid_df.to_csv('iid_results.csv', 
                  columns=['iid_prob','accuracy'], index=False)
    
    rects_df = csv_df[csv_df['num_rects'] != -1]
    rects_df = rects_df.sort_values(by=['num_rects','rect_size'])
    rects_df.to_csv('rects_results.csv', 
                    columns=['num_rects','rect_size','accuracy'], index=False)
    
    