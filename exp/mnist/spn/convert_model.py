"""

Script to convert SPN learned by Poon, Domingos method [1] (using their Java implemntation) 
to format readable by the implementation of Zhao et al, [2] (C++)

[1] Sum-Product Networks: A New Deep Architecture ,  UAI 2011
http://spn.cs.washington.edu/spn/

[2] A Unified Approach for Learning the Parameters of Sum-Product Networks, NIPS 2016
http://www.cs.cmu.edu/%7Ehzhao1/papers/ICML2016/spn_release.zip


"""



import os
import pandas as pd
import numpy as np
import argparse
import re
from xml.dom import minidom

INPUT_DIM_1 = 28
INPUT_DIM_2 = 28
UNUSED_NODE_WEIGHT = 0.01
VAR_VARIANCE = 1.0
DCMP_DELIM = '@' 
CLASSIFIER_MODE = False
BASE_CLASS_NODE_ID = 1000000
LABEL_NODE_BASE_ID = 1000100
PROD_NODE_BASE_ID = 1001000
K = 10
ROOT_NODE_ID = (BASE_CLASS_NODE_ID + K + 1) if CLASSIFIER_MODE else 0



pn_re = re.compile(r'(<)(\d+) (\d+) (\d+) (\d+)(>)')


def get_edges(id1,target_ids, weights=[]):
    edges = []    
    for i,t_id in enumerate(target_ids):
        if weights:
            assert(len(weights) == len(target_ids))
            edges.append(get_edge_row(id1,t_id,weight=weights[i]))
        else:
            edges.append(get_edge_row(id1,t_id))
    return pd.DataFrame(edges)

def get_edge_row(id1,id2,weight=-1):
    edge_row = pd.Series({'id1': id1, 'id2': id2})
    if weight != -1:
        edge_row['weight'] = weight
        edge_row['str_rep'] = '%d,%d,%f' % (edge_row['id1'],edge_row['id2'],edge_row['weight'])
    else:
        edge_row['str_rep'] = '%d,%d' % (edge_row['id1'],edge_row['id2'])
        
    return edge_row
    
def get_node_row(n_id,n_type,prnts):
    node_row = pd.Series({'id': n_id, 'type': n_type, 'prnts': prnts})
    node_row['str_rep'] = '%d,%s' % (node_row['id'],node_row['type'])
    return node_row
    
def create_classifier_nodes(K,root_node_row):
    classifier_nodes = []
    label_nodes = []
    prod_nodes = []
    for k in range(K):
        copy_row = root_node_row.copy()
        copy_row['id'] = BASE_CLASS_NODE_ID + (k)
        copy_row['prnts'] = 1
        copy_row['label'] = k
        copy_row['str_rep'] = '%d,%s' % (copy_row['id'],copy_row['type'])
        classifier_nodes.append(copy_row)
        pn = get_node_row(PROD_NODE_BASE_ID+k,'PRD',1)
        pn['label'] = k
        prod_nodes.append(pn)
    
    prod_nodes_df = pd.DataFrame(prod_nodes)
    classifier_nodes_df = pd.DataFrame(classifier_nodes)
    
    new_root_row = root_node_row.copy()
    new_root_row['id'] = ROOT_NODE_ID
    new_root_row['prnts'] = 0
    new_root_row['str_rep'] = '%d,%s' % (new_root_row['id'],new_root_row['type'])
    
    new_root_df = pd.DataFrame([new_root_row])
    
    
    for k in range(K):
        label_node_t = { 'id': LABEL_NODE_BASE_ID + k, 'type': 'BINNODE', 'prnts': 1,
                        'var_id': INPUT_DIM_1*INPUT_DIM_2 + (k),'T': 1.0, 'F':0.0 ,  'label': k}
        label_node_t['str_rep'] = '%d,%s,%d,%f,%f' % (label_node_t['id'],label_node_t['type'],label_node_t['var_id'],
                    label_node_t['F'],label_node_t['T'])
        
        label_node_f = { 'id': LABEL_NODE_BASE_ID + k + 10, 'type': 'BINNODE',
                        'prnts': 1, 'var_id': INPUT_DIM_1*INPUT_DIM_2 + (k),'T': 0.0, 'F':1.0, 'label': k }
        label_node_f['str_rep'] = '%d,%s,%d,%f,%f' % (label_node_f['id'],label_node_f['type'],label_node_f['var_id'],
                    label_node_f['F'],label_node_f['T'])
        
        label_nodes.append(pd.Series(label_node_t))
        label_nodes.append(pd.Series(label_node_f))
    
    label_nodes_df = pd.DataFrame(label_nodes)
    
    
    all_nodes_df = pd.concat((new_root_df,prod_nodes_df,classifier_nodes_df,label_nodes_df))
    return all_nodes_df

def create_classifier_edges(cn_df,orig_root_edges_df):
    orig_root_w = list(orig_root_edges_df['weight'])
    cls_edges = pd.DataFrame()
    # create root edges
    for k in range(K):
        target_id = cn_df[(cn_df['type'] == 'SUM') & (cn_df['label'] == k)]['id'].max()
        f_labels = list(cn_df[(cn_df['type'] == 'BINNODE') & (cn_df['label'] != k)
        & (cn_df['F'] == 1)]['id'])
        t_label = list(cn_df[(cn_df['type'] == 'BINNODE') & (cn_df['label'] == k)
        & (cn_df['T'] == 1)]['id'])
        
        pr_edges = get_edges(cn_df[(cn_df['type'] == 'PRD') & (cn_df['label'] == k)]['id'].max(), 
                                   target_ids= [target_id] + f_labels + t_label )
        
        
        cls_edges = pd.concat((pr_edges,cls_edges,get_edges(target_id,list(orig_root_edges_df['id2']),weights=orig_root_w)))
        
    new_root_edges = get_edges(ROOT_NODE_ID,list(cn_df[(cn_df['type'] == 'PRD') & (cn_df['label'] < K)]['id']), [(1.0/K)]*K)
    return pd.concat((new_root_edges,cls_edges))
    
    
def get_abs_id(regions_dict,reg_idx,idx):
    region = regions_dict[reg_idx]
    if region['type'] == 'coarse':
        return region['sum_nodes'][idx]['abs_id']
    else:
        return region['abs_id'][idx]
    
def parse_prod_node(prod_node_str):
    prod_node = {}
    regs = [int(x) for x in prod_node_str.strip('@').strip().split(' ')]
    sum_nodes = [(regs[0],regs[2]),(regs[1],regs[3])]
    prod_node['type'] = 'PRD'
    prod_node['chds'] = sum_nodes
    return prod_node
    
def parse_sum_node(sum_node_str):
    sum_node = {}
    if DCMP_DELIM in sum_node_str:
        sp_str = sum_node_str.split(':')
        sum_node['type'] = 'SUM'
        sum_node['cnt'] = float(sp_str[0])
        un_weights = []
        prod_nodes = []
        for i in np.arange(1,len(sp_str),step=2):
            prod_nodes.append(parse_prod_node(sp_str[i]))
            un_weights.append(float(sp_str[i+1]))
        sum_node['chds'] = prod_nodes
        # Originally weights are un-normalized, so we normalize them here
        sum_node['weights'] = np.array(un_weights) / np.sum(un_weights)
    return sum_node

def region_to_id(a1,a2,b1,b2):
    region_id = ((a1*INPUT_DIM_1+a2-1)*INPUT_DIM_2+b1)*INPUT_DIM_2+b2-1
    return region_id

def load_region(region_xml):
    region = {}
    if 'MEAN' in [node.nodeName for node in region_xml.childNodes]:
        region['type'] = 'fine'
        region['means'] = [float(x) for x in region_xml.childNodes[3].childNodes[0].data.strip().split(' ')[2:]]
        region['sum_nodes'] = []
        
    else:
        region['type'] = 'coarse'   
        sum_nodes_str = region_xml.childNodes[1].childNodes[0].data.strip().split('\n')
        region['sum_nodes'] = [parse_sum_node(x) for x in sum_nodes_str[1:]]
        
    coords = [int(x) for x in region_xml.childNodes[0].data.strip().split(' ')]
    a1,a2,b1,b2 = coords
    region['id'] = region_to_id(a1,a2,b1,b2)
    region['coords'] = coords
  
    return region

def convert_and_parse_xml(src_model_fname):
    dst_model_fname = os.path.basename(src_model_fname).split('.')[0] + '.xml.mdl'
    with open(dst_model_fname, 'wb') as wfile:
        wfile.write('<MODEL>\n')
        with open(src_model_fname, 'rb') as rfile:
            for line in rfile.readlines():
                newline = line
                if '<CNT>' in line:
                    newline = line.strip() + '</CNT>'
                elif '<MEAN>' in line:
                    newline = line.strip() + '</MEAN>'
                    
                elif pn_re.findall(line):
                    newline = pn_re.sub(r'@ \2 \3 \4 \5 @',line)
                
                wfile.write(newline.strip() + os.linesep)
            wfile.write('</MODEL>\n')
            
    xmldoc = minidom.parse(dst_model_fname)
    os.remove(dst_model_fname)
    return xmldoc
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        help="Path to model definitions generated using Java code.",
        type=str
    )
    parser.add_argument(
        "outname",
        help="Filename to save output to.",
        type=str
    )
    args = parser.parse_args()

    
    variables = []
    nodes = []
    edges = []
    region_dict_list = []
    abs_id_map = {}
    regions_dict = {}
    cnt = 0
    
    
    src_model_fname = args.model_path
    dst_model_fname = args.outname
    
    print "Parsing raw model file..."
    xmldoc = convert_and_parse_xml(src_model_fname)
    region_list = xmldoc.getElementsByTagName('REGION')
    for region_node in region_list:
        region = load_region(region_node)
        regions_dict[region['id']] = region
        region_dict_list.append(region)
        if region['type'] == 'fine':
            cnt += 1

    nc = 0 # global node counter
    region_dict_list.reverse()
    print "Collecting variable and sum nodes..."
    for region in region_dict_list:
        if region['type'] == 'coarse':
            for j,sn in enumerate(region['sum_nodes']):
                if sn:
                    node_row = {'id': nc,'address': (region['id'],j),'type':'SUM'} 
                    node_row['str_rep'] = '%d,%s' % (nc,node_row['type'])
                    nodes.append(node_row)
                    sn['abs_id'] = nc
                    nc += 1
        else: # fine.
            region['abs_id'] = []
            var_idx = region['coords'][0]*INPUT_DIM_2 + region['coords'][2]
            for i,m in enumerate(region['means']):
                node_row = { 'id': nc, 'var_id': var_idx, 'mean': m, 'type':'NORMALNODE' }
                node_row['str_rep'] = '%d,%s,%d,%f,%f' % (nc,node_row['type'],node_row['var_id'],
                    node_row['mean'],VAR_VARIANCE)
                nodes.append(node_row)
                region['abs_id'].append(nc)
                nc += 1
    
    print "Collecting product nodes..."             
    for region in region_dict_list:
        if region['type'] == 'coarse':
            for j,sn in enumerate(region['sum_nodes']):
                if sn:
                    for pn in sn['chds']:
                        pn_chds = [get_abs_id(regions_dict,reg_idx,idx) for (reg_idx,idx) in pn['chds']]
                        node_row = { 'id':nc, 'chds': pn_chds, 'type':'PRD'}
                        node_row['str_rep'] = '%d,%s' % (nc,node_row['type'])
                        nodes.append(node_row)
                        pn['abs_id'] = nc
                        nc += 1
        
    nodes_df = pd.DataFrame(nodes) 
    nodes_df['prnts'] = 0
    
    
    print "Collecting edges..."
    edges = []
    sum_nodes_df = nodes_df[nodes_df['type'] == 'SUM']
    for i,sn_row in sum_nodes_df.iterrows():
        reg_id, idx = sn_row['address']
        sn = regions_dict[reg_id]['sum_nodes'][idx]
        for i,chd in enumerate(sn['chds']):
            sum_edge_row = {'id1': sn['abs_id'], 'id2': chd['abs_id'], 'weight': sn['weights'][i] }
            nodes_df.set_value(chd['abs_id'],'prnts',1)
            sum_edge_row['str_rep'] = '%d,%d,%f' % (sum_edge_row['id1'],sum_edge_row['id2'],sum_edge_row['weight'])
            edges.append(sum_edge_row)
            
    prd_nodes_df = nodes_df[nodes_df['type'] == 'PRD']
    for i,pn_row in prd_nodes_df.iterrows():
        for j,chd in enumerate(pn_row['chds']):
            prd_edge_row = {'id1': pn_row['id'], 'id2': chd } 
            nodes_df.set_value(chd,'prnts',1)
            prd_edge_row['str_rep'] = '%d,%d' % (prd_edge_row['id1'],prd_edge_row['id2'])
            edges.append(prd_edge_row)
    
    edges_df = pd.DataFrame(edges)
    
    if CLASSIFIER_MODE:
        # Add nodes and edges used in classification mode
        classifier_nodes_df = create_classifier_nodes(K,nodes_df.loc[0])
        root_edges = edges_df[edges_df['id1']== 0 ]
        cls_edges = create_classifier_edges(classifier_nodes_df,root_edges)
        edges_df = pd.concat((cls_edges,edges_df))
        nodes_df = pd.concat((classifier_nodes_df,nodes_df))

    # Remove nodes with no parents unless it's the root node   
    valid_nodes_df = nodes_df[(nodes_df['prnts'] > 0) | (nodes_df['id'] == ROOT_NODE_ID)]
    
    if CLASSIFIER_MODE:
        # remove old root and connected edges
        valid_nodes_df = valid_nodes_df[(valid_nodes_df['id'] != 0)] 
        edges_df = edges_df[edges_df['id1'] != 0 ]
        
    print "Exporting SPN in spn_opt format..."
    with open(dst_model_fname,'wb') as f_out:
        f_out.write('#NODES\n')
        f_out.writelines([x+os.linesep for x in list(valid_nodes_df['str_rep'])])
        f_out.write('#EDGES\n')
        f_out.writelines([x+os.linesep for x in list(edges_df['str_rep'])])
    
    
    print "Done!"