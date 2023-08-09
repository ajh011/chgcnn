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

#generate initial x encoding (atom number and atom_init)
def struc2nodeattrs(struc, import_feat: bool = False, directory: str = "cif"):
    node_attrs = []
    for site in struc.sites:
        for el in site.species:
            node_attrs.append(el.Z)
    if import_feat == True:
        with open(osp.join(directory,'atom_init.json')) as atom_init:
            atom_vecs = json.load(atom_init)
            node_attrs = [atom_vecs[f'{z}'] for z in node_attrs]

    return node_attrs


#generate hypergraph dictionary elements for singleton sets
def struc2singletons(struc,  hgraph = [], tol=0.01, import_feat: bool = False, directory: str = "cif"):
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
def struc2pairs(struc, hgraph, nbr_lst, radius = 4.5, gauss_dim: int = 35):

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
def struc2triplets(struc, hgraph, nbr_lst, gauss_dim = 35):

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
def hgraph_gen(struc, cell = False):
    hgraph = []
    nbr_lst,reformat_nbr_lst = get_nbrlist(struc)
    #hgraph = struc2singletons(struc, hgraph, directory= dir)
    hgraph = struc2pairs(struc, hgraph, nbr_lst)
    #hgraph = struc2triplets(struc, hgraph, reformat_nbr_lst)
    #hgraph = struc2motifs(struc, hgraph, reformat_nbr_lst)
    if cell == True:
        hgraph = struc2cell(struc, hgraph)
    
    return hgraph



## Defining hypergraph generating functions
def hgraph2hgraph(hgraph_package):
    he_index = [[],[]]
    he_attrs=[]
    running_index = 0
    for hedge in hgraph_package:
        he_attrs.append(hedge[3])
        for hedge_neighbor in hedge[2]:
            he_index[1].append(running_index)
            he_index[0].append(hedge_neighbor)
        running_index += 1
    return he_index, he_attrs

## Returns hedge/node adjaceney matrix for BONDS ONLY ATM!!!!!
def node_hedge_pack(hedges, num_nodes=12, orders=['bond']):
    hedge_packer = []
    for i in hedges:
        nodes = i[2]
        weight = torch.tensor(1/len(nodes))
        weights = weight.repeat(len(nodes))
        nodes = torch.tensor(nodes)
        hots = torch.sparse.FloatTensor(nodes.unsqueeze(0), weights, (num_nodes,))
        hedge_packer.append(hots)

    hots = torch.stack(hedge_packer, dim =0)
    
    return hots
    
    
## Defining pyg hypergraph gen
def gen_pyghypergraph(hgraph, node_attrs):
    graph = Data()

    num_nodes = torch.tensor(node_attrs).shape[0]

    he_index, he_attrs = hgraph2hgraph(hgraph)
    node_hedge_adj = node_hedge_pack(hgraph, num_nodes=num_nodes)
    num_hedges = he_index[1][-1]

    graph.x = torch.tensor(node_attrs).float()
    graph.hyperedge_index = torch.tensor(he_index).long()
    graph.hyperedge_attr = torch.tensor(he_attrs).float()
    graph.node_hedge_adj = node_hedge_adj
    graph.num_hedges = he_index[1][-1]
    graph.num_nodes = torch.tensor(node_attrs).shape[0]

    return graph



##Build data structure in form of (vanilla) pytorch dataset (not PytorchGeometric!)
class CrystalHypergraphDataset(Dataset):
    def __init__(self, cif_dir, dataset_ratio=1.0, radius=4.0, n_nbr=12):
        super().__init__()

        self.radius = radius
        self.n_nbr = n_nbr

        self.cif_dir = cif_dir

        with open(f'{cif_dir}/id_prop_band_form_hull.csv') as id_prop:
            id_prop = csv.reader(id_prop)
            self.id_prop_data = [row for row in id_prop]

    def __len__(self):
        return len(self.id_prop_data)
    
    def __getitem__(self, index, report = True):
        mp_id, band, form_en, en_hull = self.id_prop_data[index]
        crystal_path = osp.join(self.cif_dir, mp_id)
        crystal_path = crystal_path + '.cif'
        if report == True:
            start = time.time()
        struc = CifParser(crystal_path).get_structures()[0]
        hgraph = hgraph_gen(struc, cell=False)
        node_attrs = struc2nodeattrs(struc, directory='/mnt/data/ajh', import_feat=True)
        hgraph = gen_pyghypergraph(hgraph,node_attrs)
        hgraph.y = torch.tensor(float(form_en), dtype = torch.float)
        if report == True:
            duration = time.time()-start
            print(f'Processed {mp_id} in {round(duration,5)} sec')
        return {
            'hgraph': hgraph,
            'mp_id' : mp_id
            }
    
class InMemoryCrystalHypergraphDataset(Dataset):
    def __init__(self, data_dir, csv_dir = ''):
        super().__init__()

        if csv_dir == '':
            csv_dir = data_dir

        self.csv_dir = csv_dir
        self.data_dir = data_dir

        with open(osp.join(csv_dir, 'ids.csv')) as id_file:
            ids_csv = csv.reader(id_file)
            ids = [mp_id[0] for mp_id in ids_csv]
            self.ids = ids
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        mp_id = self.ids[index]
        file_dir = osp.join(self.data_dir, mp_id + '_hg.pt')
        data = torch.load(file_dir)
 

        return data
    

def process_data(idx):
    with open(f'dataset/ids.csv','a') as ids:
        try:
            d = dataset[idx]
            torch.save(d['hgraph'], 'dataset/{}_hg.pt'.format(d['mp_id']))
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
    dataset = CrystalHypergraphDataset('/mnt/data/ajh')
    with open(f'dataset/ids.csv','w') as ids:
        print('Clearing id list in dataset/ids.csv')
    run_process()
