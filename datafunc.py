import numpy as np
import math

import os
import os.path as osp
import csv
import json
import itertools
import time

from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import \
    LocalStructOrderParams, \
    VoronoiNN, \
    CrystalNN, \
    JmolNN, \
    MinimumDistanceNN, \
    MinimumOKeeffeNN, \
    EconNN, \
    BrunnerNN_relative, \
    MinimumVIRENN

import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset


from multiprocessing import Pool
from tqdm import tqdm

#generate custom neighbor list to be used by all struc2's with nearest neighbor determination technique as parameter
def get_nbrlist(struc, nn_strategy = 'mind', max_nn=12):
    NN = {
        # these methods consider too many neighbors which may lead to unphysical resutls
        'voro': VoronoiNN(tol=0.2),
        'econ': EconNN(),
        'brunner': BrunnerNN_relative(),

        # these two methods will consider motifs center at anions
        'crys': CrystalNN(),
        'jmol': JmolNN(),

        # not sure
        'minokeeffe': MinimumOKeeffeNN(),

        # maybe the best
        'mind': MinimumDistanceNN(),
        'minv': MinimumVIRENN()
    }

    nn = NN[nn_strategy]

    center_idxs = []
    neighbor_idxs = []
    offsets = []
    distances = []

    reformat_nbr_lst = []

    for n in range(len(struc.sites)):
        neigh = []
        neigh = [neighbor for neighbor in nn.get_nn(struc, n)]

        neighbor_reformat=[]
        for neighbor in neigh[:max_nn-1]:
            neighbor_index = neighbor.index
            offset = struc.frac_coords[neighbor_index] - struc.frac_coords[n] + neighbor.image
            m = struc.lattice.matrix
            offset = offset @ m
            distance = np.linalg.norm(offset)
            center_idxs.append(n)
            neighbor_idxs.append(neighbor_index)
            offsets.append(offset)
            distances.append(distance)

            neighbor_reformat.append((neighbor_index, offset, distance))
        reformat_nbr_lst.append((n,neighbor_reformat))
    nbr_list = [center_idxs, neighbor_idxs, offsets, distances]

    return nbr_list, reformat_nbr_lst


#generate hypergraph dictionary elements for singleton sets
def struc2singletons(struc,  hgraph = [], tol=0.01, import_feat: bool = True, directory: str = "cif"):
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
        with open(osp.join(directory,'atom_init.json')) as atom_init:
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
        self.steps = steps-1
        
    def expand(self, distance, sig=None, tolerance = 0.01):
        drange = self.dmax-self.dmin
        step_size = drange/self.steps
        if sig == None:
            sig = step_size/2
        ds = [self.dmin + i*step_size for i in range(self.steps+1)]
        expansion = [math.exp(-(distance-center)**2/(2*sig**2))  for center in ds]
        expansion = [i if i > tolerance else 0 for i in expansion]
        return expansion
    
#Add bond nodes to hgraph list
def struc2pairs(struc, hgraph, nbr_lst, radius = 4.5, gauss_dim: int = 24):

    pair_center_idx = nbr_lst[0]
    pair_neighbor_idx = nbr_lst[1]
    distances = nbr_lst[3]
    
    if gauss_dim != 1:
        ge = gaussian_expansion(dmin = 0, dmax = radius, steps = gauss_dim)
            
    bond_index = 0
    ## currently double counts pair-wise edges/makes undirected edges
    for pair_1,pair_2,dist in zip(pair_center_idx, pair_neighbor_idx,distances):
        if gauss_dim != 1:
            dist = ge.expand(dist)
        hgraph.append(['bond', bond_index, [pair_1, pair_2], dist])
        bond_index += 1
    
    return hgraph



#Add ALIGNN-like triplets with angle feature vector
def struc2triplets(struc, hgraph, nbr_lst, gauss_dim = 10):

    #requires second output of get_nbr_lst!!!!! leaving this as reminder
    reformat_nbr_lst = nbr_lst
    
    if gauss_dim != 1:
        ge = gaussian_expansion(dmin = -1, dmax = 1, steps = gauss_dim)
    
    triplet_index = 0
    for cnt_idx, neighborset in reformat_nbr_lst:
            for i in itertools.combinations(neighborset, 2):
                (pair_1_idx, offset_1, distance_1), (pair_2_idx, offset_2, distance_2) = i

                offset_1 = np.stack(offset_1)
                offset_2 = np.stack(offset_2)
                cos_angle = (offset_1 * offset_2).sum(-1) / (np.linalg.norm(offset_1, axis=-1) * np.linalg.norm(offset_2, axis=-1))

                #Stop-gap to fix nans from zero displacement vectors
                cos_angle = np.nan_to_num(cos_angle, nan=1)
                
                if gauss_dim != 1:
                    cos_angle = ge.expand(cos_angle)
            
                hgraph.append(['triplet', triplet_index, [cnt_idx, pair_1_idx, pair_2_idx], cos_angle])
                triplet_index += 1

    return hgraph

#Types of structure-order parameters to calculate
all_types = [ "cn",
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
def struc2motifs(struc, hgraph, nbr_lst, types = all_types, lsop_tol = 0.05):
    reformat_nbr_lst = nbr_lst

    neighborhoods = []
    ####IMPORTANT: REQUIRES YOU RUN struc2atoms FIRST####
    for n, neighborset in reformat_nbr_lst:
        neigh_idxs = []
        for i in neighborset:
            neigh_idxs.append(i[0])
        neighborhoods.append([n, neigh_idxs])
        
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
def hgraph_gen(struc, dir = 'cif', cell = False):
    hgraph = []
    nbr_lst,reformat_nbr_lst = get_nbrlist(struc)
    hgraph = struc2singletons(struc, hgraph, directory= dir)
    hgraph = struc2pairs(struc, hgraph, nbr_lst)
    hgraph = struc2triplets(struc, hgraph, reformat_nbr_lst)
    hgraph = struc2motifs(struc, hgraph, reformat_nbr_lst)
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

def decompose(hgraph, order_types=['atom','bond','triplet','motif']):
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
    atoms, bonds, triplets, motifs = decompose(hgraph)
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

    atom_trip_het = [[],[]]
    for atom in atoms:
        for triplet in triplets:
            if contains(triplet[2],atom[2]):
                atom_trip_het[0].append(atom[1])
                atom_trip_het[1].append(triplet[1])
    edges['atom','in','triplet'] = atom_trip_het

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

    bond_trip_het = [[],[]]
    for bond in bonds:
        for triplet in triplets:
            if contains(triplet[2],bond[2]):
                bond_trip_het[0].append(bond[1])
                bond_trip_het[1].append(triplet[1])
    edges['bond','in','triplet'] = bond_trip_het

    bond_motifs_het = [[],[]]
    for bond in bonds:
        for motif in motifs:
            if contains(motif[2],bond[2]):
                bond_motifs_het[0].append(bond[1])
                bond_motifs_het[1].append(motif[1])
    edges['bond','in','motif'] = bond_motifs_het


    trip_trip_hom = [[],[]]
    for t1 in triplets:
        for t2 in triplets:
            if t1!= t2:
                if touches(t1, t2):
                    trip_trip_hom[0].append(t1[1])
                    trip_trip_hom[1].append(t2[1])
    edges['triplet', 'touches', 'triplet'] = trip_trip_hom

    trip_motifs_het = [[],[]]
    for triplet in triplets:
        for motif in motifs:
            if contains(motif, triplet):
                trip_motifs_het[0].append(triplet[1])
                trip_motifs_het[1].append(motif[1])
    edges['triplet', 'in', 'motif'] = trip_motifs_het

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

##Form pytorch geometric HeteroData (heterograph) from hgraph
def pyg_heterodata(hgraph, undirected = True):
    graph = HeteroData()
    atoms, bonds, triplets, motifs = decompose(hgraph)
    rel_edges = hetero_rel_edges(hgraph)

    graph['atom'].x = torch.stack([torch.tensor(atom[3],dtype = torch.float) for atom in atoms],dim=0)
    graph['bond'].x = torch.tensor(np.array([bond[3] for bond in bonds]),dtype = torch.float)
    graph['triplet'].x = torch.tensor(np.array([triplet[3] for triplet in triplets]),dtype = torch.float)
    graph['motif'].x =torch.tensor(np.array([motif[3] for motif in motifs]),dtype = torch.float)
    #graph['cell'].x = torch.rand([1,64], dtype = torch.float)
    
    graph['atom', 'bonds', 'atom'].edge_index = torch.tensor(rel_edges['atom', 'bonds', 'atom']).long()
    graph['atom', 'in', 'bond'].edge_index =  torch.tensor(rel_edges['atom', 'in', 'bond']).long()
    graph['atom', 'in', 'triplet'].edge_index =  torch.tensor(rel_edges['atom', 'in', 'triplet']).long()
    graph['atom', 'in', 'motif'].edge_index =  torch.tensor(rel_edges['atom', 'in', 'motif']).long()
    graph['bond', 'touches', 'bond'].edge_index =  torch.tensor(rel_edges['bond', 'touches', 'bond']).long()
    graph['bond', 'in', 'triplet'].edge_index =  torch.tensor(rel_edges['bond', 'in', 'triplet']).long()
    graph['bond', 'in', 'motif'].edge_index =  torch.tensor(rel_edges['bond', 'in', 'motif']).long()
    graph['triplet', 'touches', 'triplet'].edge_index =  torch.tensor(rel_edges['triplet', 'touches', 'triplet']).long() 
    graph['triplet', 'in', 'motif'].edge_index =  torch.tensor(rel_edges['triplet', 'in', 'motif']).long()
    graph['motif', 'touches', 'motif'].edge_index =  torch.tensor(rel_edges['motif', 'touches', 'motif']).long()

    #orders = ['motif', 'atom', 'bond']
    #for string in orders:
    #    graph['cell', 'contains', string].edge_index = torch.tensor(rel_edges['cell', 'contains', string]).long()
    #    if undirected == True:
            ####FORM UNDIRECTED EDGES FROM UNMATCHED PAIRS####
    #        graph[string, 'in', 'cell'].edge_index = torch.stack([torch.tensor(rel_edges['cell','contains',string][i]) for i in [1,0]], dim=0).long()
    if undirected == True:
        graph['motif','contains','atom'].edge_index =torch.stack([torch.tensor(rel_edges['atom','in','motif'][i]) for i in [1,0]], dim=0).long()
        graph['motif','contains','bond'].edge_index =torch.stack([torch.tensor(rel_edges['bond','in','motif'][i]) for i in [1,0]], dim=0).long()
        graph['motif','contains','triplet'].edge_index =torch.stack([torch.tensor(rel_edges['triplet','in','motif'][i]) for i in [1,0]], dim=0).long()
        graph['triplet','contains','atom'].edge_index =torch.stack([torch.tensor(rel_edges['atom','in','triplet'][i]) for i in [1,0]], dim=0).long()
        graph['triplet','contains','bond'].edge_index =torch.stack([torch.tensor(rel_edges['bond','in','triplet'][i]) for i in [1,0]], dim=0).long()
        graph['bond','contains','atom'].edge_index =torch.stack([torch.tensor(rel_edges['atom','in','bond'][i]) for i in [1,0]], dim=0).long()
    return graph



##Build data structure in form of (vanilla) pytorch dataset (not PytorchGeometric!)
class CrystalHypergraphDataset(Dataset):
    def __init__(self, cif_dir, dataset_ratio=1.0, radius=4.0, n_nbr=12):
        super().__init__()

        self.radius = radius
        self.n_nbr = n_nbr

        self.cif_dir = cif_dir

        with open(f'{cif_dir}/id_prop.csv') as id_prop:
            id_prop = csv.reader(id_prop)
            self.id_prop_data = [row for row in id_prop]

    def __len__(self):
        return len(self.id_prop_data)
    
    def __getitem__(self, index, report = True):
        mp_id, target = self.id_prop_data[index]
        crystal_path = osp.join(self.cif_dir, mp_id)
        crystal_path = crystal_path + '.cif'
        if report == True:
            start = time.time()
        struc = CifParser(crystal_path).get_structures()[0]
        hgraph = hgraph_gen(struc, cell=False, dir=self.cif_dir)
        relgraph = pyg_heterodata(hgraph, undirected = True)
        relgraph.y = torch.tensor(float(target), dtype = torch.float)
        if report == True:
            duration = time.time()-start
            print(f'Processed {mp_id} in {round(duration,5)} sec')
        return {
            'relgraph': relgraph,
            'mp_id' : mp_id
            }
    

def process_data(idx):
    try:
        d = dataset[idx]
        torch.save(d['relgraph'], 'dataset/{}.pt'.format(d['mp_id']))
    except:
        print(f'Cannot process index {idx}')


def run_process(N=None, processes=10):
    if N is None:
        N = len(dataset)

    pool = Pool(processes)

    for _ in tqdm(pool.imap_unordered(process_data, range(N)), total=N):
        pass


if __name__ == '__main__':
    dataset = CrystalHypergraphDataset('cif')
    run_process()
