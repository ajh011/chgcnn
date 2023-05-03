import numpy as np
import math

import os
import csv
import json

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import LocalStructOrderParams, VoronoiNN

import torch
from torch_geometric.data import HeteroData

#generate hypergraph dictionary elements for singleton sets
def struc2singletons(struc,  hgraph = [], tol=0.01, import_feat: bool = True, directory: str = ""):
    singletons = struc.get_neighbor_list(r = tol, exclude_self=False)[0]
    atom_count = 0
    for node in singletons:
        hgraph.append(['atom', atom_count, [atom_count], None])
        atom_count += 1
    #extract features (Z and possibly coordinates) for atoms
    site_lst = struc.sites
    features = [[],[]]
    i = 0
    for site in site_lst:
        #### IMPORTANT: ASSOCIATES SITE INDEX FOR LATER REFERENCE IN MOTIF GENERATION #####
        site.properties = {'index': i}
        features[1].append(site.coords) #Coordinate of sites
        z_site = [element.Z for element in site.species]
        features[0].append(z_site[0]) #Atomic num of sites
        i+=1
    #import features from CGCNN atom_init file
    if import_feat == True:
        with open(f'{directory}atom_init.json') as atom_init:
            atom_vecs = json.load(atom_init)
            features[0] = [torch.tensor(atom_vecs[f'{z}']).float() for z in features[0]]
    for hedge, feature in zip(hgraph, features[0]):
        hedge[3] = feature
    return hgraph




#define gaussian expansion for distance features

class gaussian_expansion(object):
    def __init__(self, dmin, dmax, steps):
        assert dmin<dmax
        self.dmin = dmin
        self.dmax = dmax
        self.steps = steps
        
    def expand(self, distance, sig=None, tolerance = 0.01):
        drange = self.dmax-self.dmin
        step_size = drange/self.steps
        if sig == None:
            sig = step_size/2
        ds = [self.dmin + i*step_size for i in range(self.steps)]
        expansion = [math.exp(-(distance-center)**2/(2*sig**2))  for center in ds]
        expansion = [i if i > tolerance else 0 for i in expansion]
        return expansion
    
#Add bond nodes to hgraph list
def struc2pairs(struc, hgraph, radius: float = 4, min_rad: bool = False, max_neighbor: float = 13, tol: float = 2, gauss_dim: int = 24):
    nbr_lst = struc.get_neighbor_list(r = radius, exclude_self=True)

    pair_center_idx = nbr_lst[0]
    pair_neighbor_idx = nbr_lst[1]
    distances = nbr_lst[3]
    
    if gauss_dim != 1:
        ge = gaussian_expansion(dmin = 0, dmax = radius, steps = gauss_dim)
            
    features = []
    n_count=0
    bond_index = 0
    pair_1_last = pair_center_idx[0]
    ## currently double counts pair-wise edges/makes undirected edges
    for pair_1,pair_2,dist in zip(pair_center_idx, pair_neighbor_idx,distances):
        #Accounts for max_neighbor
        if pair_1 == pair_1_last:
            n_count +=1
        else:
            pair_1_last = pair_1
            n_count = 1
        if n_count < max_neighbor:
            if gauss_dim != 1:
                dist = ge.expand(dist)
            hgraph.append(['bond', bond_index, [pair_1, pair_2], dist])
            bond_index += 1
        
    return hgraph

#Types of structure-order parameters to calculate
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

#Add motif hedges to hgraph

def struc2motifs(struc, hgraph, types = types, lsop_tol = 0.05):
    neighborhoods = []
    vnn = VoronoiNN(tol=0.1, targets=None)
    ####IMPORTANT: REQUIRES YOU RUN struc2atoms FIRST####
    for n in range(len(struc.sites)):
        neigh = [neighbor.properties['index'] for neighbor in vnn.get_nn(struc, n)]
        neighborhoods.append([n, neigh])
        neigh = []
        
    lsop = LocalStructOrderParams(types)
    motif_index = 0
    for site, neighs in neighborhoods:
        ##Calculate order parameters for Voronoii neighborhood (excluding center)
        feat = lsop.get_order_parameters(struc, site, indices_neighs = neighs)
        for n,f in enumerate(feat):
            if f == None:
                feat[n] = 0
            elif f > 1:
                feat[n] = f
            ##CONSIDER INVERT LSOP SO THAT 1 corresponds to shape as opposed to 0 abs(f-1) and make 1 when 0.
            ##Account for tolerance:
            elif f > lsop_tol:
                feat[n] = f
            else:
                feat[n] = 0
        ##Add center to node set before adding to hgraph list
        neighs.append(site)
        hgraph.append(['motif', motif_index, neighs, feat])
        motif_index += 1

    return hgraph

#Include unit cell hedge for pooling
def struc2cell(struc, hgraph, random_x = True):
    if random_x == True:
        feat = np.random.rand(64)
    else:
        feat = None
    nodes = list(range(len(struc.sites)))
    hgraph.append(['cell', 0, nodes, feat])
    return hgraph

## Now bring together process into overall hgraph generation
def hgraph_gen(struc, dir = '', cell = False):
    hgraph = []
    hgraph = struc2singletons(struc, hgraph, directory= dir)
    hgraph = struc2pairs(struc, hgraph)
    hgraph = struc2motifs(struc, hgraph)
    if cell == True:
        hgraph = struc2cell(struc, hgraph)
    
    return hgraph

## Helper functions for relatives heterograph construction

def ordertype(hgraph, string):
    order_hedge = []
    for hedge in hgraph:
        if hedge[0] == string:
            order_hedge.append(hedge)
    return order_hedge    

def decompose(hgraph, order_types=['atom','bond','motif']):
    sep = []
    for string in order_types:
        sep.append(ordertype(hgraph, string))
    return sep
    
def contains(big, small):
    if all(item in big for item in small):
        return True
    else:
        return False
    
def touches(one, two):
    if any(item in one for item in two):
        return True
    else:
        return False
    
##Define function that generates relatives heterograph edge indices
def hetero_rel_edges(hgraph, cell_vector = True):
    atoms, bonds, motifs = decompose(hgraph)
    edges = {}
    atom_atom_hom = [[],[]]
    for bond in bonds:
        atom_atom_hom[0].append(bond[2][0])
        atom_atom_hom[1].append(bond[2][1])
    edges['atom','bonds','atom'] = atom_atom_hom

    atom_bonds_het = [[],[]]
    for atom in atoms:
        for bond in bonds:
            if contains(bond[2],atom[2]):
                atom_bonds_het[0].append(atom[1])
                atom_bonds_het[1].append(bond[1])
    edges['atom','in','bond'] = atom_bonds_het

    atom_motifs_het = [[],[]]
    for atom in atoms:
        for motif in motifs:
            if contains(motif[2],atom[2]):
                atom_motifs_het[0].append(atom[1])
                atom_motifs_het[1].append(motif[1])
    edges['atom','in','motif'] = atom_motifs_het

    bond_bond_hom = [[],[]]
    for bond1 in bonds:
        for bond2 in bonds:
            if bond1!=bond2:
                if touches(bond1[2], bond2[2]):
                    bond_bond_hom[0].append(bond1[1])
                    bond_bond_hom[1].append(bond2[1])

    edges['bond','touches','bond'] = bond_bond_hom


    bond_motifs_het = [[],[]]
    for bond in bonds:
        for motif in motifs:
            if contains(motif[2],bond[2]):
                bond_motifs_het[0].append(bond[1])
                bond_motifs_het[1].append(motif[1])
    edges['bond','in','motif'] = bond_motifs_het

    mot_mot_hom = [[],[]]
    for m1 in motifs:
        for m2 in motifs:
            if m1!=m2:
                if touches(m1, m2):
                    mot_mot_hom[0].append(m1[1])
                    mot_mot_hom[1].append(m2[1])

    edges['motif','touches','motif'] = mot_mot_hom

    if cell_vector == True:
        orders = ['motif', 'atom', 'bond']
        for string, order in zip(orders, decompose(hgraph, orders)):
            edge_idx = [[],[]]
            for ent in order:
                edge_idx[0].append(0)
                edge_idx[1].append(ent[1])
            edges['cell', 'contains', string] = edge_idx
    
    return edges



## Turn into useful function that just takes directory as input


def hetero_relgraph_list_from_dir(directory='cif', root='', radius:float=4.0, undirected = False):
    if root == '':
        root = os. getcwd()
    directory = root+'/'+directory
    print(f'Searching {directory} for CIF data to convert to hgraphs')
    with open(f'{directory}/id_prop.csv') as id_prop:
        id_prop = csv.reader(id_prop)
        id_prop_data = [row for row in id_prop]
    rel_graphs=[]
    hgraphs = []
    for filename, fileprop in id_prop_data:
        try:
            file = directory+'/'+filename+'.cif'
            struc = CifParser(file).get_structures()[0]
            graph = HeteroData()
            hgraph = hgraph_gen(struc, cell=True, dir=directory+'/')
            hgraphs.append(hgraph)
            rel_edges = hetero_rel_edges(hgraph)
            atoms, bonds, motifs = decompose(hgraph)
            graph['atom'].x = torch.stack([torch.tensor(atom[3],dtype = torch.float) for atom in atoms],dim=0)
            graph['bond'].x = torch.tensor(np.array([bond[3] for bond in bonds]),dtype = torch.float)
            graph['motif'].x =torch.tensor(np.array([motif[3] for motif in motifs]),dtype = torch.float)
            graph['cell'].x = torch.rand([1,64], dtype = torch.float)

            graph['atom', 'bonds', 'atom'].edge_index = torch.tensor(rel_edges['atom', 'bonds', 'atom']).long()
            graph['atom', 'in', 'bond'].edge_index =  torch.tensor(rel_edges['atom', 'in', 'bond']).long()
            graph['atom', 'in', 'motif'].edge_index =  torch.tensor(rel_edges['atom', 'in', 'motif']).long()
            graph['bond', 'touches', 'bond'].edge_index =  torch.tensor(rel_edges['bond', 'touches', 'bond']).long()
            graph['bond', 'in', 'motif'].edge_index =  torch.tensor(rel_edges['bond', 'in', 'motif']).long()
            graph['motif', 'touches', 'motif'].edge_index =  torch.tensor(rel_edges['motif', 'touches', 'motif']).long()

            orders = ['motif', 'atom', 'bond']
            for string in orders:
                graph['cell', 'contains', string].edge_index = torch.tensor(rel_edges['cell', 'contains', string]).long()
                if undirected == True:
                    ####FORM UNDIRECTED EDGES FROM UNMATCHED PAIRS####
                    graph[string, 'in', 'cell'].edge_index = torch.stack([torch.tensor(rel_edges['cell','contains',string][i]) for i in [1,0]], dim=0).long()
            if undirected == True:
                graph['motif','contains','atom'].edge_index =torch.stack([torch.tensor(rel_edges['atom','in','motif'][i]) for i in [1,0]], dim=0).long()
                graph['motif','contains','bond'].edge_index =torch.stack([torch.tensor(rel_edges['bond','in','motif'][i]) for i in [1,0]], dim=0).long()
                graph['bond','contains','atom'].edge_index =torch.stack([torch.tensor(rel_edges['atom','in','bond'][i]) for i in [1,0]], dim=0).long()

        


            graph.y = torch.tensor(float(fileprop), dtype = torch.float)
            rel_graphs.append(graph)

            print(f'Added {filename} to relgraph set')
        except:
            print(f'Error with {filename}, confirm existence')
                
    print('Done generating relatives graph data with features')
    return rel_graphs
