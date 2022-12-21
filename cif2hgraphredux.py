import numpy as np
import os
import csv
import json

from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
import torch

## CIF -> tensor([[node_index,...],[hyper_edge_index,...]]) ##

## subfunction for hedge_list responsible for singleton/node hedges, needs small tolerance
## to include itself in get_neighbor_list 

## might need to rework to include features in each subfunction

## should be passed first in hgraph construction, since it assumes to be first 

def cif2singletons(cif_file, tol=0.01, hedge_list = []):
    struc = CifParser(cif_file).get_structures()[0]
    singletons = struc.get_neighbor_list(r = tol, exclude_self=False)[0]
    if hedge_list == []:
        hedge_counter = 0
    else:
        hedge_counter = np.max(hedge_list[1]) + 1
    for node in singletons:
        hedge_list[0].append(node)
        hedge_list[1].append(hedge_counter)
        hedge_counter += 1
    return hedge_list




## CIF, tensor([[node_index,...],[hyper_edge_index,...]]) -> tensor([[node_index,...],[hyper_edge_index,...]]) ##

## subfunction for hedge_list responsible for pair-wise hedges

## might need to rework to include features in each subfunction

def cif2pairs(cif_file, hedge_list, radius: float = 3, min_rad: bool = True, tol: float = 0.1):
    struc = CifParser(cif_file).get_structures()[0]
    if min_rad == False:
        nbr_lst = struc.get_neighbor_list(r = , exclude_self=True)
    elif min_rad == True:
        nbr_lst = struc.get_neighbor_list(r = 25, exclude_self=True)
        min_rad = np.min(nbr_lst[3])
        nbr_lst = struc.get_neighbor_list(r = min_rad+tol, exclude_self=True)

    pair_center_idx = nbr_lst[0]
    pair_neighbor_idx = nbr_lst[1]
    if hedge_list == []:
        hedge_counter = 0
    else:
        hedge_counter = np.max(hedge_list[1]) + 1

    ## currently double counts pair-wise edges
    for pair_1,pair_2 in zip(pair_center_idx, pair_neighbor_idx):
        hedge_list[0].append(pair_1)
        hedge_list[0].append(pair_2)

        hedge_list[1].append(hedge_counter)
        hedge_list[1].append(hedge_counter)
        hedge_counter += 1

    return hedge_list


    last_center = pair_center_idx[0]
        new_center = pair_1
        if last_center != new_center:
            last_center = pair_1
            hedge_counter += 1

## CIF -> tensor([[node_index,...],[hyper_edge_index,...]]) ##

# takes cif file and returns array (2 x num_nodes_in_hedges) of hedge index
# (as specified in the HypergraphConv doc of PyTorch Geometric)
# found by collecting neighbors within spec radius for each node in one hedge
def cif2hyperedges(cif_file, radius: float = 3, min_rad = False, tolerance = 0.1):
    struc = CifParser(cif_file).get_structures()[0]
    ##Determines minimum radius and returns neighbor list for within min radius + tolerance
    if min_rad == True:
        nbr_lst = struc.get_neighbor_list(r = 25, exclude_self=True)
        min_rad = np.min(nbr_lst[3])
        nbr_lst = struc.get_neighbor_list(r = min_rad + tolerance, exclude_self=True)
    else:
        nbr_lst = struc.get_neighbor_list(r = radius, exclude_self=True)
    edge_list = np.stack((nbr_lst[0], nbr_lst[1])).transpose()
    edge_list = torch.tensor(edge_list)

    tk = edge_list[0][0]
    hedge_index = []
    node_index = []
    for i, j in edge_list:
        if i != tk:
            hedge_index.append(tk)
            node_index.append(tk)
            tk = i
        node_index.append(j)
        hedge_index.append(i)
    node_index.append(edge_list[-1][0])
    hedge_index.append(edge_list[-1][0])
    hedge_list = torch.stack((torch.tensor(node_index), torch.tensor(hedge_index)))
    return hedge_list

def cif2hgraph(cif, radius:float = 3, min_rad = False, tolerance = 0.1):
    pos = cif2nodepos(cif)[0]
    x = cif2nodepos(cif)[1]
    hedge_indx = cif2hyperedges(cif, radius, min_rad = False, tolerance = 0.1)
    chgraph = Data(x=x, hyperedge_index=hedge_indx, pos=pos)
    return chgraph

#Returns list of hypergraph data objects (pyg data with hedge_index) from given directory

def hgraph_list_from_dir(directory='cif_data', root='', atom_vecs = True, radius:float=3.0, min_rad = False, tolerance = 0.1):
    if root == '':
        root = os. getcwd()
    directory = root+'/'+directory
    print(f'Searching {directory} for CIF data to convert to hgraphs')
    with open(f'{directory}/id_prop.csv') as id_prop:
        id_prop = csv.reader(id_prop)
        id_prop_data = [row for row in id_prop]
    hgraph_data_list = []
    if atom_vecs:
        with open(f'{directory}/atom_init.json') as atom_init:
            atom_vecs = json.load(atom_init)
            for filename, fileprop in id_prop_data:
                try:
                    file = directory+'/'+filename+'.cif'
                    graph = cif2hgraph(file, radius=radius, min_rad = False, tolerance = 0.1)
                    graph.y = torch.tensor(float(fileprop))
                    nodes_z = graph.x.tolist()
                    nodes_atom_vec = [atom_vecs[f'{z}'] for z in nodes_z]
                    graph.x = torch.tensor(nodes_atom_vec).float()
                    hgraph_data_list.append(graph)
                    print(f'Added {filename} to hgraph set')
                except:
                    print(f'Error with {filename}, confirm existence')
    else:
        for filename, fileprop in id_prop_data:
                try:
                    file = directory+'/'+filename+'.cif'
                    graph = cif2hgraph(file, radius=radius, min_rad = False, tolerance = 0.1)
                    graph.y = torch.tensor(float(fileprop))
                    hgraph_data_list.append(graph)
                    print(f'Added {filename} to hgraph set')
                except:
                    print(f'Error with {filename}, confirm existence')
    print('Done generating hypergraph data')
    return hgraph_data_list


