import torch
from lib.dataset import TrafficDataset
from lib.load_mat import load_predefined_matrix
from trainer import Trainer
from config import nyc_mb_config, nyc_mt_config, nyc_bt_config, seq_len, pred_len
from torch.utils.data import DataLoader
import argparse
from lib.metric_utils import init_seed

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='nyc_mb', type=str, help='nyc_mb,nyc_mt,nyc_bt')
    parser.add_argument('--device', default='cuda:0', type=str, help='')
    parser.add_argument('--model', default='CausalMTF', type=str, help='')
    parser.add_argument('--seed', default=12, type=int, help='')
    parser.add_argument('--batch_size', default=32, type=int, help='')
    parser.add_argument('--num_epochs', default=350, type=int, help='run epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='')
    parser.add_argument('--interval_1', default=60, type=int, help='minutes')
    parser.add_argument('--interval_2', default=60, type=int, help='minutes')
    parser.add_argument('--m2n_drop_percent', default=0.3, type=float)
    parser.add_argument('--n2m_drop_percent', default=0.3, type=float)
    
    args = parser.parse_args()
    

    init_seed(args.seed)
    batch_size = args.batch_size
    interval_1 = args.interval_1
    interval_2 = args.interval_2
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    device = (args.device if torch.cuda.is_available() else 'cpu')

    # data load
    train_set = TrafficDataset(args.dataset, interval_1, interval_2, seq_len=seq_len, pred_len=pred_len, mode='train', device = device)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    val_set = TrafficDataset(args.dataset, interval_1, interval_2, seq_len=seq_len, pred_len=pred_len, mode='val', device = device)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    test_set = TrafficDataset(args.dataset, interval_1, interval_2, seq_len=seq_len, pred_len=pred_len, mode='test', device = device)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    if args.dataset == 'nyc_mb':
        model_conf = nyc_mb_config[args.model]
    elif args.dataset == 'nyc_mt':
        model_conf = nyc_mt_config[args.model]
    elif args.dataset == 'nyc_bt':
        model_conf = nyc_bt_config[args.model]

    model_conf['m2n_drop_percent'] = args.m2n_drop_percent
    model_conf['n2m_drop_percent'] = args.n2m_drop_percent

    predefined_adj =  torch.from_numpy(load_predefined_matrix(args.dataset, model_conf["pred_mat"])).to(device)

    trainer = Trainer(predefined_adj = predefined_adj, model_args=model_conf).to(device)

    nParams = sum([p.nelement() for p in trainer.network.parameters()])
    print('Number of model parameters is', nParams)

    # model training
    trainer.train_exec(num_epochs, lr, weight_decay, train_loader, val_loader, test_loader, device)

    trainer.test_exec(test_loader, device)

    _, xm_test_loss_mae, xn_test_loss_mae, _, xm_test_loss_rmse, xn_test_loss_rmse = trainer.test_exec_val(test_loader, device)

    print('Test Phase: mode1 Mae Loss: %.4f, mode2 Mae Loss: %.4f,' % (xm_test_loss_mae, xn_test_loss_mae))
    print('Test Phase: mode1 Rmse Loss: %.4f, mode2 Rmse Loss: %.4f,' % (xm_test_loss_rmse, xn_test_loss_rmse))