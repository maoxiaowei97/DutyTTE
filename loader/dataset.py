import pickle
from torch.utils.data import Dataset
from os.path import join
from loader.node2vec import get_node2vec
import os
def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    return file
ws =  get_workspace()

class TrajFastDataset(Dataset):
    def __init__(self, city, path, device, is_pretrain):
        super().__init__()
        name = city
        self.device = device

        shrink_G_path = join(path, f"{name}_shrink_G.pkl")
        shrink_A_path = join(path, f"{name}_shrink_A.ts")
        shrink_NZ_path = join(path, f"{name}_shrink_nodeindex_dict.pkl")

        print("loading")
        self.G = pickle.load(open(shrink_G_path, "rb"))
        self.A = pickle.load(open(shrink_A_path, "rb"))
        self.shrink_nonzero_dict = pickle.load(open(shrink_NZ_path, "rb"))
        self.n_vertex = len(self.G.nodes)
        print('loaded')

        if is_pretrain:
            embed_path = join(path, f"{city}_node2vec.pkl")
            path_path = join(path, f"{city}_path.pkl")
            get_node2vec(self.G, embed_path, path_path)
            print('node2vec finished')
