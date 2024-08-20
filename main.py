from os.path import join
import torch
from loader.dataset import TrajFastDataset
from utils.argparser import get_argparser
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    parser = get_argparser()
    args = parser.parse_args()
    save_time = f'checkpoint_t{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

    if args.device == "default":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(device)

    dataset = TrajFastDataset(args.d_name, args.path, device, is_pretrain=False)
    n_vertex = dataset.n_vertex
    print(f"vertex: {n_vertex}")

    with open(join(args.model_path, f"{args.model_name}.info"), "w") as f:
        f.writelines(str(args))

    if args.method == "drl_path_prediction":

        from path_prediction.policy_network import Planner
        from path_prediction.trainer import Trainer

        model = Planner(dataset.G, dataset.A, device, args, pretrain_path=None)
        trainer = Trainer(model, dataset, device, args)
        trainer.drl_path_prediction(args.n_epoch, args.lr)

    if args.method == "drl_path_inference":

        from path_prediction.policy_network import Planner
        from path_prediction.trainer import Trainer

        model = Planner(dataset.G, dataset.A, device, args, pretrain_path=None)
        trainer = Trainer(model, dataset, device, args)
        trainer.load_model_for_inference()

    if args.method == 'MoEUQ':
        from uncertainty_quantification.trainer import Trainer
        from uncertainty_quantification.MoEUQ import MoEUQ_network
        model = MoEUQ_network(args).to(device)

        suffix = "cd"
        trainer = Trainer(model, dataset, device, args)
        trainer.generated_path_eta_uq(args)

    if args.method == "calibration":
        from uncertainty_quantification.calibration import CalibrateModel
        from uncertainty_quantification.MoEUQ import MoEUQ_network
        model = MoEUQ_network(args).to(device)
        calibrator = CalibrateModel(model, dataset, device, args)
        calibrator.calibrate(args)