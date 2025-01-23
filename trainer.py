import torch
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math
from model.model import CausalMTF
from lib.metric_utils import masked_mean_absolute_error, get_log_dir, \
    masked_root_mean_squared_error, masked_mse_loss, dynamic_weight_average

class Trainer(nn.Module):
    def __init__(self, predefined_adj,  model_args):
        super(Trainer, self).__init__()

        self.network = CausalMTF(predefined_adj, model_args)
        self.network_init()
        self.model_name = str(type(self.network).__name__)
        self.hidden_dim = model_args.get('hidden_dim')
        self.loss_fun_rmse = masked_root_mean_squared_error
        self.loss_fun_mae = masked_mean_absolute_error
        self.save_path = model_args.get('save_path')
        self.loss_fun = masked_mean_absolute_error
        self.train_progress = []
        self.val_progress = []


    def network_init(self):
        for param in self.network.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

    def get_optimizer(self, lr, wd):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def get_scheduler(self, optim, lr_decay_step, lr_decay_rate):
        return torch.optim.lr_scheduler.StepLR(optim,
                                               step_size=lr_decay_step,
                                               gamma=lr_decay_rate)

    def forward(self, x_m, x_n, time_x, weather_x):
        """
        :param x: (B, seq_len, N, 2 * num_feats)
        :return: output_xm, output_xn
        """
        return self.network(x_m, x_n, time_x, weather_x)

    def train_exec(self, num_epochs, lr, wd, train_loader, val_loader, test_loader, device):
        print('Training on', device)
        best_epoch = 0
        best_val_loss = np.inf
        optimizer = self.get_optimizer(lr=lr, wd=wd)

        log_dir = get_log_dir(self.save_path)
        if os.path.isdir(log_dir) == False :
            os.makedirs(log_dir, exist_ok=True)
        loss_t_1 = None
        loss_t_2 = None
        for e in range(num_epochs):
            self.train()
            epoch_loss = []
            xm_epoch_loss = []
            xn_epoch_loss = []

            weight = dynamic_weight_average(loss_t_2, loss_t_1)
            print(weight)
            start_time_1 = time.time()
            for batch_id, (input_xm, true_ym, input_xn, true_yn, time_x, weather_x) in enumerate(train_loader):
                input_xm, true_ym, input_xn, true_yn = input_xm.to(device), true_ym.to(device), input_xn.to(device), true_yn.to(device)
                output_xm, output_xn, _, _ = self.forward(input_xm, input_xn, time_x, weather_x)
                
                true_ym, true_yn = true_ym, true_yn

                xm_loss = self.loss_fun(output_xm, true_ym)
                xn_loss = self.loss_fun(output_xn, true_yn)
                
                if batch_id % 100 == 0:
                    print("epochs {} batch{} finished".format(e+1, batch_id))

                loss = weight[0] * xm_loss + weight[1] * xn_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                xm_epoch_loss.append(xm_loss.item())
                xn_epoch_loss.append(xn_loss.item())
                
            start_time_2 = time.time()
            epoch_loss = np.mean(epoch_loss).item()
            xm_epoch_loss = np.mean(xm_epoch_loss).item()
            xn_epoch_loss = np.mean(xn_epoch_loss).item()
            loss_t_2 = loss_t_1
            loss_t_1 = [xm_epoch_loss, xn_epoch_loss]

            self.train_progress.append(epoch_loss)
            
            print('Train Epoch: %s/%s, Loss: %.4f, mode1Loss: %.4f, mode2Loss: %.4f, time: %.2fs' %
                  (e + 1, num_epochs, epoch_loss, xm_epoch_loss, xn_epoch_loss, start_time_2 - start_time_1))

            val_loss, xm_val_loss, xn_val_loss = self.val_exec(val_loader, device)
            print('VAL Phase: Loss %.4f, mode1Loss: %.4f, mode2Loss: %.4f,' % (val_loss, xm_val_loss, xn_val_loss))
            self.val_progress.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                test_loss_mae, xm_test_loss_mae, xn_test_loss_mae, test_loss_rmse, xm_test_loss_rmse, xn_test_loss_rmse = self.test_exec_val(test_loader, device)
                print('Test Phase: Mae Loss %.4f, mode1 Mae Loss: %.4f, mode2 Mae Loss: %.4f,' % (test_loss_mae, xm_test_loss_mae, xn_test_loss_mae))
                print('Test Phase: Rmse Loss %.4f, mode1 Rmse Loss: %.4f, mode2 Rmse Loss: %.4f,' % (test_loss_rmse, xm_test_loss_rmse, xn_test_loss_rmse))
                best_test_loss_xm = xm_test_loss_mae
                best_test_loss_xn = xn_test_loss_mae
                best_epoch = e
                self.save_model(e, lr,  log_dir, best_test_loss_xm, best_test_loss_xn)
           

        self.save_progress(num_epochs, lr)
        print('Best Epoch:', best_epoch, 'Val Loss:', best_val_loss)
    

    def val_exec(self, val_loader, device):
        self.eval()
        with torch.no_grad():
            batch_loss = []
            xm_batch_loss = []
            xn_batch_loss = []
            
            s1 = time.time()
            for batch_id, (input_xm, true_ym, input_xn, true_yn, time_x, weather_x) in enumerate(val_loader):
                input_xm, true_ym, input_xn, true_yn = input_xm.to(device), true_ym.to(device), input_xn.to(device), true_yn.to(device)
                
                output_xm, output_xn, _, _ = self.forward(input_xm, input_xn, time_x, weather_x)
                
                true_ym, true_yn = true_ym, true_yn

                xm_loss = self.loss_fun(output_xm, true_ym)
                xn_loss = self.loss_fun(output_xn, true_yn)
                
                loss = xm_loss + xn_loss

                batch_loss.append(loss.item())
                xm_batch_loss.append(xm_loss.item())
                xn_batch_loss.append(xn_loss.item())
                
            s2 = time.time()
            val_loss = np.mean(batch_loss).item()
            xm_val_loss = np.mean(xm_batch_loss).item()
            xn_val_loss = np.mean(xn_batch_loss).item()
            
            print('Inference Time: {:.4f} secs'.format(s2-s1),)
        return val_loss, xm_val_loss, xn_val_loss

    def test_exec_val(self, test_loader, device):
        self.eval()
        preds_xm, targets_xm = [], []
        preds_xn, targets_xn = [], []
        with torch.no_grad():
            for batch_id, (input_xm, true_ym, input_xn, true_yn, time_x, weather_x) in enumerate(test_loader):
                input_xm, true_ym, input_xn, true_yn = input_xm.to(device), true_ym.to(device), input_xn.to(device), true_yn.to(device)
                
                output_xm, output_xn, _, _ = self.forward(input_xm, input_xn, time_x, weather_x)
                true_ym, true_yn = true_ym, true_yn
                preds_xm.append(output_xm)
                targets_xm.append(true_ym)
                preds_xn.append(output_xn)
                targets_xn.append(true_yn)
        
        preds_xm = torch.cat(preds_xm, dim=0)
        targets_xm = torch.cat(targets_xm, dim=0)
        preds_xn = torch.cat(preds_xn, dim=0)
        targets_xn = torch.cat(targets_xn, dim=0)

        xm_loss_mae = self.loss_fun_mae(preds_xm, targets_xm).item()
        xn_loss_mae = self.loss_fun_mae(preds_xn, targets_xn).item()
        xm_loss_rmse = self.loss_fun_rmse(preds_xm, targets_xm).item()
        xn_loss_rmse = self.loss_fun_rmse(preds_xn, targets_xn).item()
        loss_mae = xm_loss_mae + xn_loss_mae
        loss_rmse = xm_loss_rmse + xn_loss_rmse
        
        return loss_mae, xm_loss_mae, xn_loss_mae, loss_rmse, xm_loss_rmse,  xn_loss_rmse
    
    def test_exec(self, test_loader, device):
        self.eval()
        preds_xm, targets_xm = [], []
        preds_xn, targets_xn = [], []
        I_m2n_graphs, I_n2m_graphs = [], []

        with torch.no_grad():
            for batch_id, (input_xm, true_ym, input_xn, true_yn, time_x, weather_x) in enumerate(test_loader):
                input_xm, true_ym, input_xn, true_yn = input_xm.to(device), true_ym.to(device), input_xn.to(device), true_yn.to(device)
                output_xm, output_xn, I_m2n, I_n2m = self.forward(input_xm, input_xn, time_x, weather_x)

                preds_xm.append(output_xm)
                targets_xm.append(true_ym)
                preds_xn.append(output_xn)
                targets_xn.append(true_yn)
                I_m2n_graphs.append(I_m2n)
                I_n2m_graphs.append(I_n2m)

        preds_xm = torch.cat(preds_xm, dim=0)
        targets_xm = torch.cat(targets_xm, dim=0)
        preds_xn = torch.cat(preds_xn, dim=0)
        targets_xn = torch.cat(targets_xn, dim=0)
        I_m2n_graphs = torch.cat(I_m2n_graphs, dim=0)
        I_n2m_graphs = torch.cat(I_n2m_graphs, dim=0)

        return preds_xm, targets_xm, preds_xn, targets_xn, I_m2n_graphs, I_n2m_graphs

    def save_model(self, epoch, lr, log_dir, best_test_loss_xm, best_test_loss_xn):
        prefix = log_dir
        file_marker = self.model_name + '_lr' + str(lr) + '_e' + str(epoch + 1) + "_"+ str(best_test_loss_xm) + "_" + str(best_test_loss_xn)
        model_path = time.strftime(prefix +'/'+ '%m%d_%H_%M_' + file_marker + '.pth')
        torch.save(self.state_dict(), model_path)
        print('save parameters to file: %s' % model_path)

    def load_model(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))
        print('load parameters from file: %s' % model_path)

    def save_progress(self, epoch, lr, ):
        prefix = 'logs/'
        file_marker = self.model_name + '_lr' + str(lr) + '_e' + str(epoch)
        log_path = time.strftime(prefix + '%m%d_%H_%M_' + file_marker + '.npy')
        np.save(log_path, np.array((self.train_progress, self.val_progress)))
        print('save log to file: %s' % log_path)

    def load_progress(self, log_path):
        total_progress = np.load(log_path)
        self.train_progress = total_progress[0]
        self.val_progress = total_progress[1]

    def plot_loss_curve(self):
        fig, ax = plt.subplots()
        x = np.arange(1, len(self.train_progress) + 1)
        ax.plot(x, self.train_progress, label='Training Loss')
        ax.plot(x, self.val_progress, label='Val Loss')
        ax.legend()
        ax.set_xlabel('Num of Epochs')
        ax.set_ylabel('Value of Loss')
        plt.savefig("train_loss_4.png")
