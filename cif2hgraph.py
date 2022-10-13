import numpy as np
import os
import csv
import json

from pymatgen.io.cif import CifWriter, CifParser
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
import torch


## CIF -> tuple(tensor([[node_index,connected_node_index],...]), tensor([dist,...])) ##

#takes cif file and returns array (2 x num_edges) of edge index
#found by collecting neighbors within radius, also adjoins distance
#associated with each edge in tuple
def cif2graphedges(cif_file, radius:float=3):
    struc = CifParser(cif_file).get_structures()[0]
    nbr_lst = struc.get_neighbor_list(radius, exclude_self=True)
    edge_list=np.stack((nbr_lst[0],nbr_lst[1])).transpose()
    edge_list=torch.tensor(edge_list)
    edge_list_w_dist = (edge_list,torch.tensor(nbr_lst[3]))
    return edge_list_w_dist



## CIF -> tuple(tensor([[node_pos], ... ]),tensor([node_atomic_num,...])) ##

#takes cif file and returns tuple of a tensor of node positions and a tensor
# of nodes atomic number, indexed same as cif2graphedges
def cif2nodepos(cif_file):
    struc = CifParser(cif_file).get_structures()[0]
    site_lst = struc.sites
    nodepos_lst = []
    nodespec_lst = []
    for site in site_lst:
        nodepos_lst.append(site.coords) #Coordinate of sites
        z_site = [element.Z for element in site.species]
        nodespec_lst.append(z_site) #Atomic number list of site species (should always be single element list for crystal)
    nodepos_arr = np.array(nodepos_lst, dtype=float)
    nodespec_arr = np.squeeze(nodespec_lst)
    return  (torch.tensor(nodepos_arr),torch.tensor(nodespec_arr))


## CIF file -> torch_geometric graph ##

#takes cif file (and optional radius for neighbor search)
#as input and returns torch_geometric graph with distances as edge_attr
#node positions from cif file and graph edges for all neighbors within radius
def cif2graph(cif_file, radius:float=3):
    pos=cif2nodepos(cif_file)[0]
    x=cif2nodepos(cif_file)[1] 
    edge_index=cif2graphedges(cif_file, radius=radius)[0]
    edge_attr=cif2graphedges(cif_file, radius=radius)[1]
    cgraph=Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr, x=x)
    return cgraph


## CIF dir w/ id_prop. csv, cif files, atom_init.json (optional)
## -> PyG dataset 

##takes cif directory with contents listed above and returns a list of pytorch geometric
##data objects with graph.y inherited from id_prop.csv and 
##atom features replaced with those inherited from atom_init.json (optional)
def graph_list_from_cif_dir(directory='cif_data', root='', atom_vecs = True, radius:float = 3):
    if root == '':
        root = os. getcwd()
    directory = root+'\\'+directory
    print(f'Searching {directory} for CIF data to convert to graphs')
    with open(f'{directory}\\id_prop.csv') as id_prop:
        id_prop = csv.reader(id_prop)
        id_prop_data = [row for row in id_prop]
    graph_data_list = []
    if atom_vecs:
        with open(f'{directory}\\atom_init.json') as atom_init:
            atom_vecs = json.load(atom_init)
            for filename, fileprop in id_prop_data:
                try:
                    file = directory+'\\'+filename+'.cif'
                    graph = cif2graph(file, radius=radius)
                    graph.y = torch.tensor(float(fileprop))
                    nodes_z = graph.x.tolist()
                    nodes_atom_vec = [atom_vecs[f'{z}'] for z in nodes_z]
                    graph.x = torch.tensor(nodes_atom_vec).float()
                    graph_data_list.append(graph)
                    print(f'Added {filename} to graph set')
                except:
                    print(f'Error with {filename}, confirm existence')
    else:
        for filename, fileprop in id_prop_data:
                try:
                    file = directory+'\\'+filename+'.cif'
                    graph = cif2graph(file, radius=radius)
                    graph.y = torch.tensor(float(fileprop))
                    graph_data_list.append(graph)
                    print(f'Added {filename} to graph set')
                except:
                    print(f'Error with {filename}, confirm existence')
    print('Done generating graph data')            
    return graph_data_list







## CIF -> tensor([[node_index,...],[hyper_edge_index,...]]) ##

# takes cif file and returns array (2 x num_nodes_in_hedges) of hedge index
# (as specified in the HypergraphConv doc of PyTorch Geometric)
# found by collecting neighbors within spec radius for each node in one hedge
def cif2hyperedges(cif_file, radius: float = 3):
    struc = CifParser(cif_file).get_structures()[0]
    nbr_lst = struc.get_neighbor_list(radius, exclude_self=True)
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

def cif2hgraph(cif, radius:float = 3):
    pos = cif2nodepos(cif)[0]
    x = cif2nodepos(cif)[1]
    hedge_indx = cif2hyperedges(cif, radius)
    chgraph = Data(x=x, hyperedge_index=hedge_indx, pos=pos)
    return chgraph

#Returns list of hypergraph data objects (pyg data with hedge_index) from given directory

def hgraph_list_from_dir(directory='cif_data', root='', atom_vecs = True, radius:float=3.0):
    if root == '':
        root = os. getcwd()
    directory = root+'\\'+directory
    print(f'Searching {directory} for CIF data to convert to hgraphs')
    with open(f'{directory}\\id_prop.csv') as id_prop:
        id_prop = csv.reader(id_prop)
        id_prop_data = [row for row in id_prop]
    hgraph_data_list = []
    if atom_vecs:
        with open(f'{directory}\\atom_init.json') as atom_init:
            atom_vecs = json.load(atom_init)
            for filename, fileprop in id_prop_data:
                try:
                    file = directory+'\\'+filename+'.cif'
                    graph = cif2hgraph(file, radius=radius)
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
                    file = directory+'\\'+filename+'.cif'
                    graph = cif2hgraph(file, radius=radius)
                    graph.y = torch.tensor(float(fileprop))
                    hgraph_data_list.append(graph)
                    print(f'Added {filename} to hgraph set')
                except:
                    print(f'Error with {filename}, confirm existence')
    print('Done generating hypergraph data')
    return hgraph_data_list


