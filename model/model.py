import torch 
import torch.nn as nn
from model.decompose import *
from model.causal import *
from model.stnet import *

class CausalMTF(nn.Module):
    def __init__(self,predefined_adj, args):
        super(CausalMTF, self).__init__()
        self.predefined_mat = predefined_adj
        self.dilation = args.get('dilation')
        self.input_dim = args.get('input_dim')
        self.time_dim = args.get('time_dim')
        self.weather_dim = args.get('weather_dim')
        self.hidden_dim = args.get('hidden_dim')
        self.emb_dim = args.get('emb_dim')
        self.spa_dim = self.hidden_dim
        self.tem_dim = self.hidden_dim
        self.Kt = args.get('Kt') 
        self.spatial_first = args.get('spatial_first') 
        self.num_of_xm = args.get('num_nodes_xm')
        self.num_of_xn = args.get('num_nodes_xn')
        self.len_input_xm = int(args.get('seq_len') * 60 // args.get('interval_xm'))
        self.len_input_xn = int(args.get('seq_len') * 60 // args.get('interval_xn'))
        self.pred_len_xm =int(args.get('pred_len') * 60 // args.get('interval_xm'))
        self.pred_len_xn = int(args.get('pred_len') * 60 // args.get('interval_xn'))
        self.m2n_drop_percent = float(args.get('m2n_drop_percent'))
        self.n2m_drop_percent = float(args.get('n2m_drop_percent'))
        self.save_path = args.get('save_path')
        if self.save_path == 'nyc_mb':
            self.scaling = 1.5
        elif self.save_path == 'nyc_mt':
            self.scaling = 5.
        elif self.save_path == 'nyc_bt':
            self.scaling = 3.
        else:
            self.scaling = 1.
        
        
        self.encoder_intra_xm = Encoder_Intra(input_dim = self.input_dim, hidden_dim = self.hidden_dim)
        self.encoder_inter_xm = Encoder_Inter(input_dim = self.input_dim, hidden_dim = self.hidden_dim)
        self.encoder_intra_xn = Encoder_Intra(input_dim = self.input_dim, hidden_dim = self.hidden_dim)
        self.encoder_inter_xn = Encoder_Inter(input_dim = self.input_dim, hidden_dim = self.hidden_dim)

        self.causal_flow_trans = Causal_FlowTrans(
            predefined_adj = self.predefined_mat,
            hidden_dim = self.hidden_dim,
            num_of_xm = self.num_of_xm,
            num_of_xn = self.num_of_xn,
            len_input_xm = self.len_input_xm,
            len_input_xn = self.len_input_xn,
            input_dim = self.input_dim,
            time_dim = self.time_dim,
            weather_dim = self.weather_dim,
            num_relations = self.predefined_mat.shape[0],
            scaling=self.scaling
        )

        self.gate_fusion_xm = GatedFusion(in_channels = self.hidden_dim, out_channels = self.hidden_dim)
        self.gate_fusion_xn = GatedFusion(in_channels = self.hidden_dim, out_channels = self.hidden_dim)

        self.xm_Es = nn.Parameter(torch.randn(self.num_of_xm, self.emb_dim))
        self.xm_Ed = nn.Parameter(torch.randn(self.num_of_xm, self.emb_dim))
        self.xn_Es = nn.Parameter(torch.randn(self.num_of_xn, self.emb_dim))
        self.xn_Ed = nn.Parameter(torch.randn(self.num_of_xn, self.emb_dim))

        if self.spatial_first:
            self.head_dim = self.spa_dim
            self.tail_dim = self.tem_dim
        else:
            self.head_dim = self.tem_dim
            self.tail_dim = self.spa_dim
        
        self.num_blocks = len(self.Kt)
        self.block_lens_xm = [self.len_input_xm]
        self.block_lens_xn = [self.len_input_xn]
        [self.block_lens_xm.append(self.block_lens_xm[-1] - (self.Kt[i] - 1) * self.dilation[i])
         for i in range(self.num_blocks)
        ]
        [self.block_lens_xn.append(self.block_lens_xn[-1] - (self.Kt[i] - 1) * self.dilation[i])
         for i in range(self.num_blocks)
        ]
        print(self.block_lens_xm)
        
        print(self.num_blocks, 'Blocks :', self.num_blocks)

        self.blocks_xm = nn.ModuleList(
            [STBlock(self.hidden_dim, self.tem_dim, self.spa_dim, self.Kt[0], self.dilation[0],
                     self.block_lens_xm[0], self.spatial_first)]
        )
        self.blocks_xm.extend(
            [STBlock(self.tail_dim, self.tem_dim, self.spa_dim, self.Kt[i], self.dilation[i],
                     self.block_lens_xm[i], self.spatial_first)
             for i in range(1, self.num_blocks)]
        )
        
        self.blocks_xn = nn.ModuleList(
            [STBlock(self.hidden_dim, self.tem_dim, self.spa_dim, self.Kt[0], self.dilation[0],
                     self.block_lens_xn[0], self.spatial_first)]
        )
        self.blocks_xn.extend(
            [STBlock(self.tail_dim, self.tem_dim, self.spa_dim, self.Kt[i], self.dilation[i],
                     self.block_lens_xn[i], self.spatial_first)
             for i in range(1, self.num_blocks)]
        )

        self.xm_fuse_layer = nn.Parameter(torch.randn(self.num_blocks * self.tail_dim, 8 * self.tail_dim))
        self.xn_fuse_layer = nn.Parameter(torch.randn(self.num_blocks * self.tail_dim, 8 * self.tail_dim))

        self.xm_output_layer = nn.Parameter(torch.randn(8 * self.tail_dim, self.pred_len_xm * self.input_dim))
        self.xn_output_layer = nn.Parameter(torch.randn(8 * self.tail_dim, self.pred_len_xn * self.input_dim))

    def forward(self, Xm, Xn, time_x, weather_x):
        """
        :param Xm: (batch, seq_len_c, num_nodes_c, in_dim)
        :param Xn: (batch, seq_len_b, num_nodes_b, in_dim)
        :return:
        """ 

        # Decomposition of mode m
        f_mi = self.encoder_intra_xm(Xm)
        f_mo = self.encoder_inter_xm(Xm)

        # Decomposition of mode n
        f_ni = self.encoder_intra_xn(Xn)
        f_no = self.encoder_inter_xn(Xn)

        # Learn Causal Flow Transition Representation and Probability Graph
        z_n2m, z_m2n, I_n2m, I_m2n = self.causal_flow_trans(
            Xm, Xn, f_mo, f_no, time_x, weather_x, self.m2n_drop_percent, self.n2m_drop_percent
        )

        # Gating Mechanism
        z_m = self.gate_fusion_xm(f_mi, z_n2m)
        z_n = self.gate_fusion_xn(f_ni, z_m2n)

        # Dynamic Graph for mode m
        m2m_mat = cal_adaptive_matrix(self.xm_Es, self.xm_Ed)
        # Dynamic Graph for mode n
        n2n_mat = cal_adaptive_matrix(self.xn_Es, self.xn_Ed)
        

        # Collaborative Forecasting: Learning spatio-temporal dependency for multi-modal forecasting
        h_m = z_m.permute(0,3,1,2)  # (batch_size, in_dim, seq_len, num_nodes)
        h_n = z_n.permute(0,3,1,2)  # (batch_size, in_dim, seq_len, num_nodes)

        h_m_skip_connections = []
        h_n_skip_connections = []
        for i in range(self.num_blocks):
            h_m = self.blocks_xm[i](h_m, m2m_mat)
            h_n = self.blocks_xn[i](h_n, n2n_mat)
            h_m_skip = h_m[:, :, -1, :]
            h_n_skip = h_n[:, :, -1, :]

            h_m_skip_connections.append(h_m_skip)  # (B, F, T, N)
            h_n_skip_connections.append(h_n_skip)

        h_m_skip_feats = torch.cat(h_m_skip_connections, dim=1).permute(0, 2, 1)
        h_n_skip_feats = torch.cat(h_n_skip_connections, dim=1).permute(0, 2, 1)

        h_m = torch.relu(h_m_skip_feats.matmul(torch.relu(self.xm_fuse_layer)))
        h_n = torch.relu(h_n_skip_feats.matmul(torch.relu(self.xn_fuse_layer)))

        y_m = torch.matmul(h_m, self.xm_output_layer).unsqueeze(1).reshape(-1, self.pred_len_xm, self.num_of_xm, self.input_dim)
        y_n = torch.matmul(h_n, self.xn_output_layer).unsqueeze(1).reshape(-1, self.pred_len_xn, self.num_of_xn, self.input_dim)

        return y_m, y_n, I_m2n, I_n2m
