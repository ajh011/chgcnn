from hetero_data import hetero_relgraph_list_from_dir
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


data_loc = 'hetero_relgraph_list.pkl'

relgraph_list = hetero_relgraph_list_from_dir()


with open(data_loc, 'wb') as storage:
    pickle.dump(relgraph_list, storage, pickle.HIGHEST_PROTOCOL)

print(f'Saved preliminary data (pre-projection) to "{data_loc}"')


del relgraph_list