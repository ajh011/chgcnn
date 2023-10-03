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
import numpy as np
import math





#generate custom neighbor list to be used by all struc2's with nearest neighbor determination technique as parameter
def get_nbrlist(struc, nn_strategy = 'voro', max_nn=12):
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


    for n in range(len(struc.sites)):
        neigh = []
        neigh = [neighbor for neighbor in nn.get_nn(struc, n)]

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

    nbr_list = [center_idxs, neighbor_idxs, offsets, distances]

    return nbr_list

#generate initial x encoding (atom number and atom_init)
def struc2nodeattrs(struc, import_feat: bool = True, directory: str = "/mnt/data/ajh"):
    node_attrs = []
    for site in struc.sites:
        for el in site.species:
            node_attrs.append(el.Z)
    if import_feat == True:
        with open(osp.join(directory,'atom_init.json')) as atom_init:
            atom_vecs = json.load(atom_init)
            node_attrs = [atom_vecs[f'{z}'] for z in node_attrs]

    return node_attrs



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
    
def struc2pairs(struc, nbr_lst, radius = 8, gauss_dim: int = 40):

    edge_index = [[],[]]
    edge_attrs = []
    pair_center_idx = nbr_lst[0]
    pair_neighbor_idx = nbr_lst[1]
    distances = nbr_lst[3]
    
    if gauss_dim != 1:
        ge = gaussian_expansion(dmin = 0, dmax = radius, steps = gauss_dim)
            


    ## currently double counts pair-wise edges/makes undirected edges
    for pair_1,pair_2,dist in zip(pair_center_idx, pair_neighbor_idx,distances):
        if gauss_dim != 1:
            dist = ge.expand(dist)

        edge_index[0].append(pair_1)
        edge_index[1].append(pair_2)
        
        edge_attrs.append(dist)
    return edge_index, edge_attrs


## Defining pyg hypergraph gen
def gen_pyggraph(struc, import_feats:bool = True, directory:str = ''):
    graph = Data()


    nbr_lst = get_nbrlist(struc) 

    node_attrs = struc2nodeattrs(struc, import_feat = import_feats, directory = directory) 
    edge_index, edge_attrs = struc2pairs(struc, nbr_lst, gauss_dim = 35)

    graph.x = torch.tensor(node_attrs).float()
    graph.edge_index = torch.tensor(edge_index).long()
    graph.edge_attr = torch.tensor(edge_attrs).float()

    return graph



##Build data structure in form of (vanilla) pytorch dataset (not PytorchGeometric!)
class CrystalGraphDataset(Dataset):
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
        mp_id, form_en = self.id_prop_data[index]
        crystal_path = osp.join(self.cif_dir, mp_id)
        crystal_path = crystal_path + '.cif'
        if report == True:
            start = time.time()
        struc = CifParser(crystal_path).get_structures()[0]
        graph = gen_pyggraph(struc, directory = self.cif_dir)
        graph.y = torch.tensor(float(form_en), dtype = torch.float)
        if report == True:
            duration = time.time()-start
            print(f'Processed {mp_id} in {round(duration,5)} sec')
        return {
            'graph': graph,
            'mp_id' : mp_id
            }
    
class InMemoryCrystalGraphDataset(Dataset):
    def __init__(self, data_dir, csv_dir = ''):
        super().__init__()

        if csv_dir == '':
            csv_dir = data_dir

        self.csv_dir = csv_dir
        self.data_dir = data_dir

        with open(osp.join(csv_dir, 'ids_cg.csv')) as id_file:
            ids_csv = csv.reader(id_file)
            ids = [mp_id[0] for mp_id in ids_csv]
            self.ids = ids
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        mp_id = self.ids[index]
        file_dir = osp.join(self.data_dir, mp_id + '_cg.pt')
        data = torch.load(file_dir)
 

        return data
    

def process_data(idx):
    with open(f'dataset/ids_cg.csv','a') as ids:
        try:
            d = dataset[idx]
            torch.save(d['graph'], 'dataset/{}_cg.pt'.format(d['mp_id']))
            ids.write(d['mp_id']+'\n')

        except:
            print(f'Cannot process index {idx}')


def run_process(N=None, processes=10):
    if N is None:
        N = len(dataset)

    pool = Pool(processes)

    for _ in tqdm(pool.imap_unordered(process_data, range(N)), total=N):
        pass


if __name__ == '__main__':
    dataset = CrystalGraphDataset('../chgcnn_old/cif')
    with open(f'dataset/ids_cg.csv','w') as ids:
        print('Clearing id list in dataset/ids_cg.csv')
    run_process()
