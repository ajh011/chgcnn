import ast
import functools
import json
import os, csv
import os.path as osp
import warnings

import numpy as np
from monty.serialization import loadfn
import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Data, Dataset
from preprocess import get_defect_structure_index

warnings.filterwarnings("ignore")


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class DefectGraphDataset(Dataset):
    def __init__(self, host_dir, defect_dir, radius=5.0, step=0.2):
        super().__init__()
        self.host_dir = host_dir
        self.defect_dir = defect_dir
        self.radius = radius
        self.ari = AtomCustomJSONInitializer('atom_init.json')
        self.gdf = GaussianDistance(dmin=0., dmax=self.radius, step=step)

        with open('dataset/C2DB/defect_name_idx.csv', 'r') as f:
            reader = csv.reader(f)
            self.data = [[row[0], ast.literal_eval(row[1]), row[2]] for row in reader]

    def __len__(self):
        return len(self.data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        defect_name, cell_batch, host_name = self.data[idx]
        defect_struc = Structure.from_file(osp.join(self.defect_dir, defect_name+'.cif'))
        host_struc = Structure.from_file(osp.join(self.host_dir, host_name+'.cif'))
        defect_graph = self.get_graph_data(defect_struc)
        defect_graph.cell_batch = torch.tensor(cell_batch, dtype=torch.long)
        host_graph = self.get_graph_data(host_struc)
        return defect_graph, host_graph

    def get_graph_data(self, struc):
        ctr_idx, nbr_idx, _, dis = struc.get_neighbor_list(self.radius, exclude_self=False)
        x = np.vstack([self.ari.get_atom_fea(site.specie.number) for site in struc])
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(np.vstack((ctr_idx, nbr_idx)), dtype=torch.long)
        edge_attr = torch.tensor(self.gdf.expand(dis), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class DefectCalcDataset(Dataset):
    def __init__(self, radius=5.0, step=0.2, task='classification'):
        self.radius = radius
        assert task in ['regression', 'classification'], 'Task must be regression or classification. Got {}'.format(task)
        self.task = task
        self.structures = loadfn('defect_data/features_defect_st.json')
        with open('defect_data/id_prop_{}.csv'.format(task), 'r') as f:
            reader = csv.reader(f)
            id_prop = [row for row in reader]

        self.id_prop = id_prop
        self.ari = AtomCustomJSONInitializer('atom_init.json')
        self.gdf = GaussianDistance(dmin=0., dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop)

    def __getitem__(self, index):
        task_id, target = self.id_prop[index]
        struc = self.structures[task_id]

        ctr_idx, nbr_idx, _, dis = struc.get_neighbor_list(self.radius, exclude_self=False)
        x = np.vstack([self.ari.get_atom_fea(site.specie.number) for site in struc])
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(np.vstack((ctr_idx, nbr_idx)), dtype=torch.long)
        edge_attr = torch.tensor(self.gdf.expand(dis), dtype=torch.float)
        if self.task == 'regression':
            target = torch.tensor([[float(target)]], dtype=torch.float)
        else:
            target = torch.tensor(int(target), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target)

if __name__ == '__main__':
    from tqdm import tqdm
    # from torch_geometric.loader import DataLoader
    #
    # dataset = DefectGraphDataset('dataset/C2DB/cifs', 'dataset/C2DB/defects')
    # loader = DataLoader(dataset, batch_size=2)
    # data = next(iter(loader))
    #
    # import csv
    #
    # to_csv = []
    # root_dir = 'dataset/C2DB/cifs'
    # for cif in tqdm(os.listdir(root_dir)):
    #     host_file = osp.join(root_dir, cif)
    #     vacs = get_defect_structure_index(host_file, 'vac')
    #
    #     for defect_struc, defect_name, cell_idx in vacs:
    #         to_csv.append([defect_name, cell_idx.tolist(), cif.split('.')[0]])
    #         # defect_struc.to('cif', 'dataset/C2DB/defects/{}.cif'.format(defect_name))
    #
    # with open('dataset/C2DB/defect_name_idx.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for row in to_csv:
    #         writer.writerow(row)

    dataset = DefectCalcDataset(task='classification')
    data = dataset[0]