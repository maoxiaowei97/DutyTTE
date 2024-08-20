import argparse
import os

def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()
def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="configure all the settings")

    # device config
    parser.add_argument("-device", type=str, help="device, [cpu, cuda]", default="default")

    # path config
    parser.add_argument("-path", type=str, help="data path", default=ws + "/processed_data")
    parser.add_argument("-model_path", type=str, help="model path", default=ws + "/model_params")
    parser.add_argument("-res_path", type=str, help="results path", default=ws + "/results")

    # data config
    parser.add_argument("-d_name", type=str, default="CityA")
    parser.add_argument("-max_decode_step", type=int, help="max_decode_step", default=51)

    # model config
    parser.add_argument("-model_name", type=str, help="model name", default="MoEUQ")
    parser.add_argument("-method", type=str, help="method: cold, naive", default="MoEUQ")
    parser.add_argument("-x_emb_dim", type=int, help="vertex embedding dim", default=100)
    parser.add_argument("-E_U", type=int, help="embedding dim in MoEUQ", default=256)
    parser.add_argument("-C", type=int, help="number of experts", default=20)
    parser.add_argument("-k", type=int, help="number of selected experts", default=5)
    parser.add_argument("-L_T", type=int, help="number of layers for transformer in path prediction", default=1)
    parser.add_argument("-m", type=int, help="number of statistical travel time", default=5)

    parser.add_argument("-n_groups", type=int, help="number of groups for group normalization", default=8)

    # confidence level
    parser.add_argument("-rho", type=int, default=0.1)
    parser.add_argument("-alpha", type=int, default=0.1)
    parser.add_argument("-delta", type=int, default=0.1)

    # training config
    parser.add_argument("-n_epoch", type=int, help="number of epoch", default=142)
    parser.add_argument("-bs", type=int, help="batch size", default=128)
    parser.add_argument("-lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("-rl_ratio", type=int, help="ratio of policy loss", default=1)
    parser.add_argument("-omega", type=int, help="ratio of policy loss", default=1)
    parser.add_argument("-beta", type=int, help="ratio of policy loss", default=50)
    parser.add_argument("-minimum_lambda", type=int, default=0)
    parser.add_argument("-maximum_lambda", type=int, default=2)
    parser.add_argument("-num_lambdas", type=int, default=200)

    # eval config
    parser.add_argument("-early_stop", type=int, help="patience of early stop", default=20)

    return parser