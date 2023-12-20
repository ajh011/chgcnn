import numpy as np
import math

import os
import os.path as osp
import json
import itertools

from robocrys.condense.fingerprint import get_structure_fingerprint

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
from torch_geometric.data import HeteroData


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
        for neighbor in sorted(neigh, key = lambda x:x[1])[:max_nn]:
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

    return reformat_nbr_lst, nn_strategy


#define gaussian expansion for feature dimensionality expansion
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
    
### Define general hyperedge type class
class HyperedgeType(object):
    def __init__(self, generate_features = True):
        self.hyperedge_index = [[],[]]
        self.hyperedge_attrs = []
        self.neighborsets = []
        self.generate_features = generate_features

### Define bonds hyperedge type for generation
class Bonds(HyperedgeType):
    def __init__(self, dir_or_nbrset=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'bond'
        self.order = 2

        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)
    
    def generate(self, dir_or_nbrset, nn_strat = 'voro', gauss_dim = 40, radius = 8):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else: 
            nbr_list = dir_or_nbrset
        self.nbrset = nbr_list
        self.nbr_strategy = nn_strat
            
        if gauss_dim != 1:
            ge = gaussian_expansion(dmin = 0, dmax = radius, steps = gauss_dim)
            
        distances = []
        bond_index = 0
        ## currently double counts pair-wise edges/makes undirected edges
        for neighbor_set in nbr_list:
            center_index = neighbor_set[0]
            for neighbor in neighbor_set[1]:
                neigh_index = neighbor[0]
                offset = neighbor[1]
                distance = neighbor[2]
            
                self.hyperedge_index[0].append(center_index)
                self.hyperedge_index[1].append(bond_index)

                self.hyperedge_index[0].append(neighbor[0])
                self.hyperedge_index[1].append(bond_index)
            
                self.neighborsets.append([center_index,neighbor[0]])
                
                distances.append(distance)
            
                bond_index += 1

            
        if self.generate_features:
            for dist in distances:
                if gauss_dim != 1:
                    dist = ge.expand(dist)
                self.hyperedge_attrs.append(dist)

### Define triplets hyperedge type for generation
class Triplets(HyperedgeType):
    def __init__(self, dir_or_nbrset=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'triplet'
        self.order = 3

        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)
    
    def generate(self, dir_or_nbrset, nn_strat = 'voro', gauss_dim = 40, radius = 8):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else:
            nbr_list = dir_or_nbrset
        if gauss_dim != 1:
            ge = gaussian_expansion(dmin = -1, dmax = 1, steps = gauss_dim)

        triplet_index = 0
        for cnt_idx, neighborset in nbr_list:
                for i in itertools.combinations(neighborset, 2):
                    (pair_1_idx, offset_1, distance_1), (pair_2_idx, offset_2, distance_2) = i

                    if self.generate_features == True:
                        offset_1 = np.stack(offset_1)
                        offset_2 = np.stack(offset_2)
                        cos_angle = (offset_1 * offset_2).sum(-1) / (np.linalg.norm(offset_1, axis=-1) * np.linalg.norm(offset_2, axis=-1))

                        #Stop-gap to fix nans from zero displacement vectors
                        cos_angle = np.nan_to_num(cos_angle, nan=1)

                        if gauss_dim != 1:
                            cos_angle = ge.expand(cos_angle)
                        
                        self.hyperedge_attrs.append(cos_angle)
                            
                    self.hyperedge_index[0].append(pair_1_idx)
                    self.hyperedge_index[1].append(triplet_index)

                    self.hyperedge_index[0].append(pair_2_idx)
                    self.hyperedge_index[1].append(triplet_index)
                    
                    self.hyperedge_index[0].append(cnt_idx)
                    self.hyperedge_index[1].append(triplet_index)
            
                    self.neighborsets.append([cnt_idx, pair_1_idx, pair_2_idx])
                    
                    triplet_index += 1


### Define bonds hyperedge type for generation
class Motifs(HyperedgeType):
    def __init__(self,  dir_or_nbrset=None, struc=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'motif'
        self.order = 12
        self.struc=struc
        
        self.all_lsop_types = [ "cn",
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
        
        if dir_or_nbrset != None:
            self.generate(dir_or_nbrset)

    def generate(self, dir_or_nbrset, nn_strat = 'mind', lsop_types = []):
        if type(dir_or_nbrset) == str:
            struc = CifParser(dir_or_nbrset).get_structures()[0]
            nbr_list, nn_strat = get_nbrlist(struc, nn_strategy = nn_strat, max_nn=12)
        else: 
            nbr_list = dir_or_nbrset 
            if self.struc == None:
                print('Structure required as input for motif neighbor lists')
            struc = self.struc

        self.nbr_strategy = nn_strat
        #for site in struc.sites:
         #   for el in site.species:
          #      node_attrs.append(el.Z)
        neighborhoods = []
        motif_index = 0
        for n, neighborset in nbr_list:
            neigh_idxs = []
            for idx in neighborset:
                neigh_idx = idx[0]
                neigh_idxs.append(neigh_idx)
                self.hyperedge_index[0].append(neigh_idx)
                self.hyperedge_index[1].append(motif_index)
            self.hyperedge_index[0].append(n)
            self.hyperedge_index[1].append(motif_index)
            neighborhoods.append([n, neigh_idxs])
            neigh_idxs.append(n)
            self.neighborsets.append(neigh_idxs)
            motif_index += 1
        if self.generate_features == True and lsop_types == []:
            lsop_types = self.all_lsop_types
        
        lsop = LocalStructOrderParams(lsop_types)
        lsop_tol = 0.05
        for site, neighs in neighborhoods:
            ##Calculate order parameters for Voronoii neighborhood (excluding center)
            #feat = lsop.get_order_parameters(struc, site, indices_neighs = neighs)
            feat = lsop.get_order_parameters(struc, site)
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
            self.hyperedge_attrs.append(feat)


### Define bonds hyperedge type for generation
class UnitCell(HyperedgeType):
    def __init__(self, dir_or_struc=None, generate_features = True):
        super().__init__(generate_features = generate_features)
        self.name = 'unit_cell'
        self.neighborsets = [[]]
        self.order = 100

        if dir_or_struc!=None:
            self.generate(dir_or_struc)

    def generate(self, dir_or_struc):
        if type(dir_or_struc) == str:
            struc = CifParser(dir_or_struc).get_structures()[0]
        else: 
            struc = dir_or_struc
        
        
        for site_index in range(len(struc.sites)):
            self.hyperedge_index[0].append(site_index)
            self.hyperedge_index[1].append(0)
            
            self.neighborsets[0].append(site_index)
            
            
        if self.generate_features:
            structure_fingerprint = get_structure_fingerprint(struc)
            self.hyperedge_attrs.append(structure_fingerprint)
        

### Helper functions for inclusion and touching criteria of hyperedges
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


### Define general crystal hypergraph class that accepts list of hyperedge types, mp_id string, and structure
class Crystal_Hypergraph(HeteroData):
    def __init__(self, struc, bonds = True, triplets = True, motifs = True, unit_cell = False,
                 mp_id: str = None, target_dict = {}, strategy = 'Aggregate'):
        super().__init__()  
        
        self.struc = struc
        self.mp_id = mp_id
        self.orders = []
        
        self.hyperedges = []
       
        if struc != None:
            ## Generate neighbor lists
            nbr_mind, _ = get_nbrlist(struc, nn_strategy = 'mind', max_nn=12)
            nbr_voro, _ = get_nbrlist(struc, nn_strategy = 'voro', max_nn=12)
        
            ## Generate bonds, triplets, motifs, and unit cell
            ## hyperedge types
            if bonds == True:
                bonds = Bonds(nbr_voro)
                self.hyperedges.append(bonds)
            if triplets == True:
                triplets = Triplets(nbr_voro)
                self.hyperedges.append(triplets)
            if motifs == True:
                motifs = Motifs(nbr_mind, struc=struc)    
                self.hyperedges.append(motifs)
            if unit_cell == True:
                unit_cell = UnitCell(struc)
                self.hyperedges.append(unit_cell)



            ## Add hyperedge types to hypergraph
            if self.hyperedges != None:
                for hyperedge_type in self.hyperedges:
                    self.add_hyperedge_type(hyperedge_type)
        
            ## Generate relatives edges and atomic info
            self.generate_atom_xs()
            self.generate_edges(strategy)
        
            ## Import target dict automatically, if passed as input of init
            if target_dict != {}:
                self.import_targets(target_dict)

    ## Function used to generate atomic features (Note these are considered hyperedge_attrs)
    def generate_atom_xs(self, import_feats=False):
        node_attrs = []
        for site in self.struc.sites:
            for el in site.species:
                node_attrs.append(el.Z)
    ## import features callsusual atom_init from CGCNN and assumes this json file 
    ## is in the current directory otherwise, feats are just atomic numbers
        if import_feats == True:
            with open('atom_init.json') as atom_init:
                atom_vecs = json.load(atom_init)
                node_attrs = [atom_vecs[f'{z}'] for z in node_attrs]
        self['atom'].hyperedge_attrs = torch.tensor(node_attrs).float()

    ## Function used to add hyperedge_type to hypergraph
    def add_hyperedge_type(self, hyperedge_type):
        self[('atom','in',hyperedge_type.name)].hyperedge_index = torch.tensor(hyperedge_type.hyperedge_index).long()
        self[hyperedge_type.name].hyperedge_attrs = torch.tensor(np.stack(hyperedge_type.hyperedge_attrs)).float()
        self.orders.append(hyperedge_type.name)

    ## Function used to determine relatives edges between different order hyperedges
    def hyperedge_inclusion(self, larger_hedgetype, smaller_hedgetype, flip = False):
        hedge_index = [[],[]]
        for small_idx, small_set in enumerate(smaller_hedgetype.neighborsets):
            for large_idx, large_set in enumerate(larger_hedgetype.neighborsets):
                if contains(large_set, small_set):
                    hedge_index[0].append(small_idx)
                    hedge_index[1].append(large_idx)
        self[(smaller_hedgetype.name, 'in', larger_hedgetype.name)].hyperedge_index = torch.tensor(hedge_index).long()
        if flip == True:
            self[(larger_hedgetype.name, 'contains', smaller_hedgetype.name)].hyperedge_index = torch.tensor([hedge_index[1],hedge_index[0]]).long()

    ## Function used to determine relatives edges between touching hyperedges of same order
    def hyperedge_touching(self, hyperedge_type):
        hedge_index = [[],[]]
        for idx_1, set_1 in enumerate(hyperedge_type.neighborsets):
            for idx_2, set_2 in enumerate(hyperedge_type.neighborsets):
                if idx_1 == idx_2: 
                    pass
                else:
                    if touches(set_1, set_2):
                        hedge_index[0].append(idx_1)
                        hedge_index[1].append(idx_2)
        self[(hyperedge_type.name, 'touches', hyperedge_type.name)].hyperedge_index = torch.tensor(hedge_index).long()
      
    ## Function used to determine labelled inter-order hyperedge relations
    def hyperedge_relations(self, larger_hedgetype, smaller_hedgetype, flip = False):
        relation_index = [[],[],[]]
        for (idx_1, nset_1), (idx_2, nset_2) in itertools.combinations(enumerate(smaller_hedgetype.neighborsets),2):
            for large_idx, large_nset in enumerate(larger_hedgetype.neighborsets):
                if contains(large_nset, nset_1) and contains(large_nset, nset_2):
                    relation_index[0].append(idx_1)
                    relation_index[1].append(large_idx)
                    relation_index[2].append(idx_2)
                    if flip == True:
                        relation_index[0].append(idx_2)
                        relation_index[1].append(large_idx)
                        relation_index[2].append(idx_1)
        self[(smaller_hedgetype.name, larger_hedgetype.name, smaller_hedgetype.name)].inter_relations_index = torch.tensor(relation_index).long()


    ## Stop-gap for atom-wise relations, at the moment
    def atom_hyperedge_relations(self, larger_hedgetype, flip = False):
        relation_index = [[],[],[]]
        for idx, nset in enumerate(larger_hedgetype.neighborsets):
            for atom_pair in itertools.combinations(nset, 2):
                relation_index[0].append(atom_pair[0])
                relation_index[1].append(idx)
                relation_index[2].append(atom_pair[1])
                if flip == True:
                    relation_index[0].append(atom_pair[0])
                    relation_index[1].append(idx)
                    relation_index[2].append(atom_pair[1])
        self[('atom', larger_hedgetype.name, 'atom')].hyperedge_relations_index = torch.tensor(relation_index).long()

    ## Function used to genertate different edge strategies (Relatives, Aggregate, Interorder, All)
    def generate_edges(self, strategy):
        if strategy == 'All':
            self.generate_relatives()
        elif strategy == 'Relatives':
            self.generate_relatives(relatives = False)
        elif strategy == 'Aggregate': 
            self.generate_relatives(touching = False, relatives = False)
        elif strategy == 'Interorder': 
            self.generate_relatives(inclusion = False, touching = False)

        
    ## Function used to generate full relatives set
    def generate_relatives(self, relatives = True, touching = True, inclusion = True, flip = True):
        if relatives & inclusion == True:
            for pair_hedge_types in itertools.permutations(self.hyperedges, 2):
                    if pair_hedge_types[0].order > pair_hedge_types[1].order:
                        self.hyperedge_inclusion(pair_hedge_types[0],pair_hedge_types[1], flip = flip)
                        self.hyperedge_relations(pair_hedge_types[0],pair_hedge_types[1])
            for hedge_type in  self.hyperedges:
                self.atom_hyperedge_relations(hedge_type)

        
        elif inclusion:
            for pair_hedge_types in itertools.permutations(self.hyperedges, 2):
                    if pair_hedge_types[0].order > pair_hedge_types[1].order:
                        self.hyperedge_inclusion(pair_hedge_types[0],pair_hedge_types[1], flip = flip)
                        
        elif relatives:
            for pair_hedge_types in itertools.permutations(self.hyperedges, 2):
                    if pair_hedge_types[0].order > pair_hedge_types[1].order:
                        self.hyperedge_relations(pair_hedge_types[0],pair_hedge_types[1])
            for hedge_type in  self.hyperedges:
                self.atom_hyperedge_relations(hedge_type)

                
        if touching:
            for hyperedge_type in self.hyperedges:
                if hyperedge_type.name == 'unit_cell':
                    pass
                else:
                    self.hyperedge_touching(hyperedge_type)
        

    ## Import targets as dictionary and save as value of heterodata
    def import_targets(self, target_dict):
        for key, value in target_dict.items():
            self[key] = value

