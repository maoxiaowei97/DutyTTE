from os.path import join
import torch
from loader.dataset import TrajFastDataset
from utils.argparser import get_argparser
import os
import numpy as np
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '1'
if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    parser = get_argparser()
    args = parser.parse_args()
    save_time = f'checkpoint_t{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

    # set device
    if args.device == "default":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
    else:
        device = torch.device(args.device)
    print(device)

    date = "2018"
    dataset = TrajFastDataset(args.d_name, [date], args.path, device, is_pretrain=True)
    n_vertex = dataset.n_vertex
    print(f"vertex: {n_vertex}")

    # before train, record the info
    with open(join(args.model_path, f"{args.model_name}.info"), "w") as f:
        f.writelines(str(args))

    if args.method == "rl_finetune":

        from planner.transformer_rl_0804 import Planner
        from planner.trainer_rl_0804 import Trainer

        model = Planner(dataset.G, dataset.A, device, rl_ratio=args.rl_ratio, max_decode_step=args.max_decode_step, x_emb_dim=100, pretrain_path='/data/maodawei/ODTUQ_0806/arranged_data_0824/chengdu_d10131415161720_h9101112131415_node2vec_2146.pkl')
        trainer = Trainer(model, dataset, device, args.early_stop, args.model_path)
        trainer.rl_finetune(args.n_epoch, args.bs, args.lr)

    if args.method == "GDP_generation":
        from planner.gdp_0807 import GDP_Planner
        from planner.gdp_trainer_0808 import Trainer

        from models_seq.seq_models import Destroyer, Restorer
        from models_seq.eps_models import EPSM

        betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
        # destroyer = Destroyer(dataset.A, betas, args.max_T, device)
        pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
        dims = eval(args.dims)

        pretrain_path = '/data/maodawei/ODTUQ_0806/arranged_data_0824/chengdu_d10131415161720_h9101112131415_node2vec_2146.pkl'
        eps_model = EPSM(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device,
                         hidden_dim=args.hidden_dim,
                         pretrain_path='/data/maodawei/ODTUQ_0806/arranged_data_0824/chengdu_d10131415161720_h9101112131415_node2vec_2146.pkl')
        diffusion_model_path = '/data/maodawei/ODTUQ_0806/sets_model/diffusion/2024-08-07 22:15:32/finished_0.pth'

        model = GDP_Planner(dataset.G, dataset.A,  device, rl_ratio=args.rl_ratio, max_decode_step=args.max_decode_step, x_emb_dim=100)
        trainer = Trainer(model, dataset, device, args.early_stop, args.model_path)
        trainer.gdp_generate(args.n_epoch, args.bs, args.lr)

    if args.method == "beam_search_generate":

        from planner.transformer_rl_0804 import Planner
        from planner.trainer_rl_0804 import Trainer

        model = Planner(dataset.G, dataset.A, device, rl_ratio=args.rl_ratio, max_decode_step=args.max_decode_step, x_emb_dim=args.x_emb_dim)
        trainer = Trainer(model, dataset, device, args.early_stop, args.model_path)
        trainer.beam_search_generate(args.n_epoch, args.bs, args.lr)

    if args.method == "mle_train":
        from planner.transformer_rl_0804 import Planner
        from planner.trainer_rl_0804 import Trainer

        model = Planner(dataset.G, dataset.A, device, rl_ratio=args.rl_ratio, max_decode_step=args.max_decode_step,
                        x_emb_dim=args.x_emb_dim)
        trainer = Trainer(model, dataset, device, args.early_stop, args.model_path)
        trainer.train(args.n_epoch, args.bs, args.lr)

    if args.method == "transformer_generate":
        from planner.transformer_plan import Planner
        from planner.trainer import Trainer

        # restorer = torch.load("/data/MaoXiaowei/ODTUQ/ODTETA/sets_model/finished_5520.pth")
        # destroyer = restorer.destroyer
        # model = Planner(dataset.G, dataset.A, restorer, destroyer, device, x_emb_dim=args.x_emb_dim, pretrain_path=pretrain_path)
        model = Planner(dataset.G, dataset.A, device, x_emb_dim=args.x_emb_dim)
        # model = torch.load("/data/MaoXiaowei/ODTUQ/ODTETA/sets_model/plan_cd_0613_checkpoint_t2024_06_13_08_23_06.pth")
        trainer = Trainer(model, dataset, device, args.model_path)
        trainer.train(args.n_epoch, args.bs, args.lr)
        model.eval()
        torch.save(model, join(args.model_path, f"{args.model_name}_{save_time}.pth"))
        # 训练基于起终点，已经经过节点作为条件的下一位置预测网络
