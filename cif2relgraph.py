import numpy as np
import os
import csv
import json

from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import LocalStructOrderParams
from torch_geometric.data import Data
import torch



## struc -> [[node_index,...],[hyper_edge_index,...]], [[hyper_edge_index,...],[[node_pos_x,y,],...],[node_atom_num,...]]##

##    or if import_feat is true:      ##
##    -> [[node_index,...],[hyper_edge_index,...]], [[hyper_edge_index,...],[[node_pos_x,y,z],...],[[node_feature_1,...],...]]##

## subfunction for hedge_list responsible for singleton/node hedges, needs small tolerance
## to include itself in get_neighbor_list 

## might need to rework to handle feat dims currently
## returns atom_num or atom_init from CGCNN for feats

## should be passed first in hgraph construction, since it assumes to be first 


def struc2singletons(struc,  hedge_list = [[],[]], tol=0.1, import_feat: bool = False, directory: str = ""):
    singletons = struc.get_neighbor_list(r = tol, exclude_self=False)[0]
    if hedge_list == [[],[]] or hedge_list == []:
        hedge_counter = 0
    else:
        hedge_counter = np.max(hedge_list[1]) + 1
    
    features = [[],[],[]]
    for node in singletons:
        hedge_list[0].append(node)
        hedge_list[1].append(hedge_counter)
        features[0].append(hedge_counter)
        hedge_counter += 1
        
    
    site_lst = struc.sites
    for site in site_lst:
        features[2].append(site.coords) #Coordinate of sites
        z_site = [element.Z for element in site.species]
        features[1].append(z_site[0]) #Atomic num of sites
    
    if import_feat == True:
        with open(f'{directory}atom_init.json') as atom_init:
            atom_vecs = json.load(atom_init)
            features[1] = [atom_vecs[f'{z}'] for z in features[1]]

    return hedge_list, features



## struc, [[node_index,...],[hyper_edge_index,...]] ##
## -> [[node_index,...],[hyper_edge_index,...]],[[hyper_edge_index,...],[[gauss_dist_exp,...],...]] ##

## subfunction for hedge_list responsible for pair-wise hedges
## can be based on threshold + min_rad or some interatomic radius

## might need to rework to include features in each subfunction
## gaussian distance for pairs

def struc2pairs(struc, hedge_list = [[],[]], radius: float = 3, min_rad: bool = True, tol: float = 0.1, gauss_dim: int = 1):
    if min_rad == False:
        nbr_lst = struc.get_neighbor_list(r = radius, exclude_self=True)
    elif min_rad == True:
        nbr_lst = struc.get_neighbor_list(r = 25, exclude_self=True)
        min_rad = np.min(nbr_lst[3])
        nbr_lst = struc.get_neighbor_list(r = min_rad+tol, exclude_self=True)

    pair_center_idx = nbr_lst[0]
    pair_neighbor_idx = nbr_lst[1]
    distances = nbr_lst[3]
    
    if hedge_list == [[],[]]:
        hedge_counter = 0
    else:
        hedge_counter = np.max(hedge_list[1]) + 1

    features = [[],[]]
    ## currently double counts pair-wise edges
    for pair_1,pair_2,dist in zip(pair_center_idx, pair_neighbor_idx,distances):
        hedge_list[0].append(pair_1)
        hedge_list[0].append(pair_2)

        hedge_list[1].append(hedge_counter)
        hedge_list[1].append(hedge_counter)
        
        features[0].append(hedge_counter)
        features[1].append(dist)
        
        hedge_counter += 1
    if gauss_dim != 1:
        ge = gaussian_expansion(dmin = 0 ,dmax = radius + 5*tol, steps = gauss_dim)
        features[1]=[ge.expand(dist) for dist in features[1]]

    
    return hedge_list, features


## Gaussian distance expansion function for pair hedge features

import math

class gaussian_expansion(object):
    def __init__(self, dmin, dmax, steps):
        assert dmin<dmax
        self.dmin = dmin
        self.dmax = dmax
        self.steps = steps
        
    def expand(self, distance, sig=None):
        drange = self.dmax-self.dmin
        step_size = drange/self.steps
        if sig == None:
            sig = step_size/2
        ds = [self.dmin + i*step_size for i in range(self.steps)]
        expansion = [math.exp(-(distance-center)**2/(2*sig**2)) for center in ds]
        return expansion





## CIF, [[node_index,...],[hyper_edge_index,...]] ##
## -> [[node_index,...],[hyper_edge_index,...]],  ##
##          [[hyper_edge_index,...],[[motif_order_param_1,...],...]] ##

## subfunction for hedge_list responsible for motif-wise hedges
## currently based only on distance (min+thresh or set range)

## might need to rework to handle output dims
## currently uses all 35 types of motif order parameters
## in LocalStructParams for features

types = [ "cn",
        "sgl_bd",
        "bent",
        "tri_plan",
        "tri_plan_max",
        "reg_tri",
        "sq_plan",
        "sq_plan_max",
        "pent_plan",
        "pent_plan_max",
        "sq",
        "tet",
        "tet_max",
        "tri_pyr",
        "sq_pyr",
        "sq_pyr_legacy",
        "tri_bipyr",
        "sq_bipyr",
        "oct",
        "oct_legacy",
        "pent_pyr",
        "hex_pyr",
        "pent_bipyr",
        "hex_bipyr",
        "T",
        "cuboct",
        "cuboct_max",
        "see_saw_rect",
        "bcc",
        "q2",
        "q4",
        "q6",
        "oct_max",
        "hex_plan_max",
        "sq_face_cap_trig_pris"]



def struc2motifs(struc, hedge_list = [[],[]], radius: float = 3, min_rad: bool = True, tol: float = 0.1):
    if min_rad == False:
        nbr_lst = struc.get_neighbor_list(r = radius, exclude_self=True)
    elif min_rad == True:
        nbr_lst = struc.get_neighbor_list(r = 25, exclude_self=True)
        min_rad = np.min(nbr_lst[3])
        nbr_lst = struc.get_neighbor_list(r = min_rad+tol, exclude_self=True)

    pair_center_idx = nbr_lst[0]
    pair_neighbor_idx = nbr_lst[1]
    if hedge_list == [[],[]]:
        hedge_counter = 0
    else:
        hedge_counter = np.max(hedge_list[1]) + 1

    last_center = pair_center_idx[0]
    neighborhoods = ([],[],[])
    neighborhood  = []

    for pair_1,pair_2 in zip(pair_center_idx, pair_neighbor_idx):
        new_center = pair_1
        if last_center != new_center:
            neighborhoods[0].append(hedge_counter)
            neighborhoods[1].append(last_pair_1)
            neighborhoods[2].append(neighborhood)
            hedge_list[0].append(last_pair_1)
            hedge_list[1].append(hedge_counter)
            last_center = pair_1
            neighborhood = []
            hedge_counter += 1
            
        hedge_list[0].append(pair_2)
        hedge_list[1].append(hedge_counter)
        neighborhood.append(pair_2)
        
        last_pair_1 = pair_1
        
    hedge_list[0].append(last_pair_1)
    hedge_list[1].append(hedge_counter)
    
    neighborhoods[0].append(hedge_counter)
    neighborhoods[1].append(last_pair_1)
    neighborhoods[2].append(neighborhood)
    
    lsop = LocalStructOrderParams(types)
    
    features = [[],[]]
    for hedge_idx, center_idx, neighbor_lst in zip(neighborhoods[0],neighborhoods[1],neighborhoods[2]):
        feature = lsop.get_order_parameters(struc, center_idx, indices_neighs = neighbor_lst)
        features[0].append(hedge_idx)
        features[1].append(np.nan_to_num(feature))
            
    return hedge_list, features



## CIF -> tensor([[node_index,...],[hyper_edge_index,...]]) ##

# takes cif file and returns array (2 x num_nodes_in_hedges) of hedge index
# (as specified in the HypergraphConv doc of PyTorch Geometric)
# found by collecting neighbors within spec radius for each node in one hedge


def cif2hedges(cif_file, radius: float = 3, min_rad: bool = True, tol: float = 0.1, features: bool = False):
    struc = CifParser(cif_file).get_structures()[0]
    hedge_list = [[],[]]
    hedge_list, singleton_feat = struc2singletons(struc, hedge_list, import_feat = True)
    hedge_list, pair_feat = struc2pairs(struc, hedge_list, radius, min_rad, tol)
    hedge_list, motif_feat = struc2motifs(struc, hedge_list, radius, min_rad, tol)
    if features == True:
        return hedge_list, (singleton_feat, pair_feat, motif_feat)
    elif features == False:
        return hedge_list



# repacks hedges in new format for easier down-stream processing

def hedge_packer(hedge_list):
    new_hedges = [[],[]]
    hedge_nodes = []
    old_hedge = hedge_list[1][0]
    for idx,(node,hedge) in enumerate(zip(hedge_list[0],hedge_list[1])):
        new_hedge = hedge_list[1][idx]
        if new_hedge != old_hedge:
            new_hedges[0].append(old_hedge)
            old_hedge = hedge_list[1][idx]
            new_hedges[1].append(hedge_nodes)
            hedge_nodes = []
        hedge_nodes.append(node)
    new_hedges[0].append(new_hedge)
    new_hedges[1].append(hedge_nodes)
    return new_hedges



# defines relatives list from hedge_pack

def relatives(hedge_pack):
    relative_list = [[],[]]
    for idx1, hedge in enumerate(hedge_pack[0]):
        #print(idx1)
        relative_list[0].append(hedge)
        relatives = []
        for idx2,nodes_contained in enumerate(hedge_pack[1]):
            #print(nodes_contained)
            if  all(item in nodes_contained for item in hedge_pack[1][idx1]) or all(item in hedge_pack[1][idx1] for item in nodes_contained):
                relatives.append(hedge_pack[0][idx2])
        relatives.remove(hedge)
        relative_list[1].append(relatives)
    return relative_list
    

# generates edge pairs for relatives graph from relatives function output

def relatives2graphedges(relative_list):
    rel_ge = [[],[]]
    for idx,pair_center in enumerate(relative_list[0]):
        for neighbor in relative_list[1][idx]:
            rel_ge[0].append(pair_center)
            rel_ge[1].append(neighbor)
    return rel_ge







## CIF -> [[node_index,...],[hyper_edge_index,...]], (singleton_feats, pairs, motifs) ##

# takes cif file and returns array (2 x num_nodes_in_hedges) of hedge index
# (as specified in the HypergraphConv doc of PyTorch Geometric)
# found by collecting neighbors within spec radius for each node in one hedge


def cif2hedges(cif_file, radius: float = 3, min_rad: bool = True, tol: float = 0.1, features: bool = False, gauss_dim: int = 5):
    struc = CifParser(cif_file).get_structures()[0]
    hedge_list = [[],[]]
    feats = []
    hedge_list, singleton_feat = struc2singletons(struc, hedge_list, import_feat = True)
    hedge_list, pair_feat = struc2pairs(struc, hedge_list, radius, min_rad, tol, gauss_dim = gauss_dim)
    hedge_list, motif_feat = struc2motifs(struc, hedge_list, radius, min_rad, tol)
    if features == True:
        return hedge_list, (singleton_feat, pair_feat, motif_feat)
    elif features == False:
        return hedge_list
    
## CIF -> array([[graph_edge_1_node_1,...],[graph_edge_1_node_2,...]]) ##


def cif2reledges(cif_file, radius: float = 3, min_rad: bool = True, tol: float = 0.1, features: bool = False, gauss_dim: int = 5):
    hedge_list, feats = cif2hedges(cif_file, radius, min_rad, tol, features, gauss_dim)
    relative = relatives(hedge_packer(hedge_list))
    edges = np.array(relatives2graphedges(relative))
    return edges, feats, hedge_list






def relgraph_list_from_dir(directory='cif', root='', atom_vecs = True, radius:float=3.0):
    if root == '':
        root = os. getcwd()
    directory = root+'\\'+directory
    print(f'Searching {directory} for CIF data to convert to hgraphs')
    with open(f'{directory}\\id_prop.csv') as id_prop:
        id_prop = csv.reader(id_prop)
        id_prop_data = [row for row in id_prop]
    relgraphs = []
    hedges = []
    feats_list = []

    for filename, fileprop in id_prop_data:
            try:
                file = directory+'\\'+filename+'.cif'
                edges, feats, hedge_list = cif2reledges(file, radius=radius, features = True)
                graph = Data()
                graph.edge_index = torch.tensor(edges, dtype = int)
                graph.y = torch.tensor(float(fileprop))
                graph.num_nodes = torch.max(graph.edge_index)
                relgraphs.append(graph)
                hedges.append(hedge_list)
                feats_list.append(feats)
                print(f'Added {filename} to relgraph set')
            except:
                print(f'Error with {filename}, confirm existence')
                
    print('Done generating relatives graph data with features')
    return relgraphs, feats_list, hedges














