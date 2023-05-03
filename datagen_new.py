from datafunc_new import hetero_relgraph_list_from_dir
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


data_loc = 'und_hetero_relgraph_list_4a.pkl'

relgraph_list = hetero_relgraph_list_from_dir(directory = 'cif', undirected = True)


with open(data_loc, 'wb') as storage:
    pickle.dump(relgraph_list, storage, pickle.HIGHEST_PROTOCOL)

print(f'Saved preliminary data (pre-projection) to "{data_loc}"')


del relgraph_list
