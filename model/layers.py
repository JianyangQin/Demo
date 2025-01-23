import torch
import torch.nn as nn 
from model.casual import *
import numpy as np
import torch.nn.functional as F
import pywt


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0., use_batchnorm=False):
        """
        Encoder for intra flow with customizable depth, batch normalization, and dropout.

        Args:
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            use_batchnorm (bool): If True, applies BatchNorm after each layer.
        """
        super().__init__()

        # Define the first layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.initial_activation = nn.ReLU()
        self.initial_dropout = nn.Dropout(dropout_rate)

        # Conditionally add batch normalization
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]) if use_batchnorm else None

        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers - 1)])
        # self.tematt = TemporalAttention()

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor (batch_size, num_of_node, sequence_len, input_dim).

        Returns:
            torch.Tensor: Encoded tensor (batch_size, num_of_node, sequence_len, hidden_dim).
        """

        x = self.initial_layer(x)
        if self.batch_norms:
            x = self.batch_norms[0](x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)

        # Hidden layers forward pass
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i + 1](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

        return x

def disentangle_intra(x, w = 'coif1', j=1):
    #print(x.shape)(32,12,266,2)
    x1 = x.detach().cpu().numpy().transpose(0,3,2,1) # [S,D,N,T]
    coef = pywt.wavedec(x1, w, level=j)
    #coefl = [coef[0]]
    #for i in range(len(coef)-1):
    #    coefl.append(None)
    coef_intra = [2*coef[0]]
    for i in range(len(coef)-1):
        coef_intra.append(coef[i+1])
    #self.xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    x1 = pywt.waverec(coef_intra, w).transpose(0,3,2,1)
    x1 = torch.from_numpy(x1).to(x.device)
    return x1

def disentangle_inter(x, w = 'coif1', j=1):
    #print(x.shape)(32,12,266,2)
    x2 = x.detach().cpu().numpy().transpose(0,3,2,1) # [S,D,N,T]
    coef = pywt.wavedec(x2, w, level=j)
    #coefl = [coef[0]]
    #for i in range(len(coef)-1):
    #    coefl.append(None)
    coef_inter = [coef[0]]
    for i in range(len(coef)-1):
        coef_inter.append(2*coef[i+1])
    #self.xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    x2 = pywt.waverec(coef_inter, w).transpose(0,3,2,1)
    x2 = torch.from_numpy(x2).to(x.device)
    return x2
        
#####################
## Flow Distanglement
#####################
# Encoder for intra flow
class Encoder_intra(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0., use_batchnorm=False):
        """
        Encoder for intra flow with customizable depth, batch normalization, and dropout.
        
        Args:
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            use_batchnorm (bool): If True, applies BatchNorm after each layer.
        """
        super().__init__()

        # Define the first layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.initial_activation = nn.ReLU()
        self.initial_dropout = nn.Dropout(dropout_rate)

        # Conditionally add batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]) if use_batchnorm else None
        
        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers - 1)])
        #self.tematt = TemporalAttention()

    def forward(self, x):
        """
        Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, num_of_node, sequence_len, input_dim).
        
        Returns:
            torch.Tensor: Encoded tensor (batch_size, num_of_node, sequence_len, hidden_dim).
        """
        # First layer forward pass
        #print(x) 
        #x0 = x
        x = disentangle_intra(x)
        #print(x-x0)
        x = self.initial_layer(x)
        if self.batch_norms:
            x = self.batch_norms[0](x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)

        # Hidden layers forward pass
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i + 1](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        #print(x.shape)
        
        return x


# Encoder for inter flow
class Encoder_inter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout_rate=0., use_batchnorm=False):
        """
        Encoder for intra flow with customizable depth, batch normalization, and dropout.
        
        Args:
            input_dim (int): Input feature dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
            num_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
            use_batchnorm (bool): If True, applies BatchNorm after each layer.
        """
        super().__init__()

        # Define the first layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.initial_activation = nn.ReLU()
        self.initial_dropout = nn.Dropout(dropout_rate)

        # Conditionally add batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]) if use_batchnorm else None
        
        # Define the hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers - 1)])
        #self.temtcn = TemporalConvNet()

    def forward(self, x):
        """
        Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, num_of_node, sequence_len, input_dim).
        
        Returns:
            torch.Tensor: Encoded tensor (batch_size, num_of_node, sequence_len, hidden_dim).
        """
        # First layer forward pass
        x = disentangle_inter(x)
        x = self.initial_layer(x)
        if self.batch_norms:
            x = self.batch_norms[0](x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)

        # Hidden layers forward pass
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i + 1](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        #x = self.temtcn(x)
        return x
 
class STDynCasualInteraction(nn.Module):
    def __init__(self, predefined_adj, hidden_dim, num_of_xm, num_of_xn, len_input_xm, len_input_xn, 
                 dilation, kt, input_dim, time_dim , weather_dim,  num_relations, spatial_first, scaling):
        super(STDynCasualInteraction, self).__init__()
        self.scaling = scaling
        self.predefined_adj = predefined_adj
        self.cf_generator = Cf_MultiRelationInteractionGeneration(
            predefined_adj, hidden_dim, num_of_xm, num_of_xn, 
            len_input_xm, len_input_xn, input_dim, time_dim, weather_dim
        )
        self.mgcn_multimodal = MultiRelationInteractionGCNLayer(num_of_xm, num_of_xn, hidden_dim, num_relations)
        
    def forward(self, xm, xn, f_mo, f_no, time_x, weather_x,
                m2n_drop_percent=0.6, n2m_drop_percent=0.6):
        """
        Forward pass for dynamic causal interaction.
        
        Args:
            z_date (torch.Tensor): Input tensor representing date features.
            z_weather (torch.Tensor): Input tensor representing weather features.
            batch_size (int): Size of the batch.
        
        Returns:
            torch.Tensor: Generated interactions.
        """
        masking_graph = self.predefined_adj.unsqueeze(0).repeat(xm.shape[0], 1, 1, 1)

        I_m2n, I_n2m  = self.cf_generator(
            xm, xn, time_x, weather_x, m2n_drop_percent, n2m_drop_percent
        )

        _I_m2n = torch.div(masking_graph * I_m2n, self.scaling)
        _I_n2m = torch.div(masking_graph * I_n2m, self.scaling)

        Z_m2n, Z_n2m = self.mgcn_multimodal(f_mo, f_no, _I_m2n, _I_n2m)
        
        Z_m2n = Z_m2n + f_no
        Z_n2m = Z_n2m + f_mo
        
        return Z_n2m, Z_m2n, I_n2m, I_m2n


class MultiRelationInteractionGCNLayer(nn.Module):
    def __init__(self, node_xm, node_xn, hidden_dim, num_relations):
        """
        Multi-Relation Interaction Graph Convolutional Layer to model cross-mode interactions with multiple relations.
        
        Args:
            node_count_mode1 (int): Number of nodes in mode 1.
            node_count_mode2 (int): Number of nodes in mode 2.
            hidden_dim (int): Output dimensionality of the interaction GCN.
            num_relations (int): Number of different relations between mode 1 and mode 2 nodes.
        """
        super(MultiRelationInteractionGCNLayer, self).__init__()
        
        # Learnable weights for combining different relations
        self.relation_weights_mode1tmode2 = nn.Parameter(torch.FloatTensor(num_relations))
        nn.init.constant_(self.relation_weights_mode1tmode2 , 1.0 / num_relations)  # Initially uniform weights
        self.relation_weights_mode2tmode1 = nn.Parameter(torch.FloatTensor(num_relations))
        nn.init.constant_(self.relation_weights_mode2tmode1, 1.0 / num_relations)  # Initially uniform weights

    def forward(self, x_m, x_n, I_m2n, I_n2m):
        """
        Forward pass for multi-relation interaction graph convolution.
        
        Args:
            x_m (torch.Tensor): Input tensor for mode 1, shape (batch_size, time_steps, node_count_mode1, hidden_dim).
            x_n (torch.Tensor): Input tensor for mode 2, shape (batch_size, time_steps, node_count_mode2, hidden_dim).
            interaction_adj (torch.Tensor): Multi-relation interaction adjacency matrix, shape (num_relations, node_count_mode1, node_count_mode2).
        
        Returns:
            torch.Tensor: Combined interaction output across multiple relations.
        """
        num_relations = I_m2n.size(1)
        
        # Collect outputs for each relation
        relation_outputs_n2m = []
        relation_outputs_m2n = []
        for r in range(num_relations):
            output_n2m = torch.einsum('blnc, bmn -> blmc', x_n, I_n2m[:,r,:,:].to(torch.float32))
            output_m2n = torch.einsum('blmc, bnm -> blnc', x_m, I_m2n[:,r,:,:].transpose(1,2).to(torch.float32))

            relation_outputs_n2m.append(output_n2m)
            relation_outputs_m2n.append(output_m2n)
        
        # Stack outputs from all relations (shape: num_relations, batch_size, time_steps, node_count_mode1, hidden_dim)
        relation_outputs_n2m = torch.stack(relation_outputs_n2m, dim=0)
        relation_outputs_m2n = torch.stack(relation_outputs_m2n, dim=0)
        # Weight and combine different relation outputs (broadcasting relation weights)
        relation_weights_m2n = self.relation_weights_mode1tmode2.view(num_relations, 1, 1, 1, 1)
        relation_weights_n2m = self.relation_weights_mode2tmode1.view(num_relations, 1, 1, 1, 1)

        Z_n2m = (relation_weights_n2m * relation_outputs_n2m).sum(dim=0)
        Z_m2n = (relation_weights_m2n * relation_outputs_m2n).sum(dim=0)
         
        return Z_m2n, Z_n2m


class Cf_MultiRelationInteractionGeneration(nn.Module):
    def __init__(self, predefined_adj, hidden_dim, num_of_xm, num_of_xn, len_input_xm, len_input_xn, input_dim, time_dim, weather_dim):
        super(Cf_MultiRelationInteractionGeneration, self).__init__()
        # Define the gating unit, here using a convolutional layer as an example
        self.predefined_adj = predefined_adj
        self.num_relations = predefined_adj.shape[0]

        self.input_dim = input_dim
        self.input_len_xm = len_input_xm
        self.input_len_xn = len_input_xn
        self.num_xm = num_of_xm
        self.num_xn = num_of_xn
        self.embed_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.conv_time = nn.Conv1d(in_channels=2, out_channels=self.hidden_dim//4, kernel_size=time_dim)
        self.conv_weather = nn.Conv1d(in_channels=2, out_channels=self.hidden_dim//4, kernel_size=weather_dim)

        self.conv_xm = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim//4, kernel_size=1)
        self.conv_xn = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim//4, kernel_size=1)

        self.m2n_date_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.m2n_weather_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.n2m_date_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.n2m_weather_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.emb_layer_mode1_add_date = nn.Linear(self.hidden_dim//2  * self.input_len_xm, self.embed_dim)
        self.emb_layer_mode2_add_date = nn.Linear(self.hidden_dim//2  * self.input_len_xn, self.embed_dim)
        self.emb_layer_mode1_add_weather = nn.Linear(self.hidden_dim//2  * self.input_len_xm, self.embed_dim)
        self.emb_layer_mode2_add_weather = nn.Linear(self.hidden_dim//2  * self.input_len_xn, self.embed_dim)

    def forward(self, xm, xn, time_x, weather_x, n2m_drop_percent=0.6, m2n_drop_percent=0.6):

        batch_size = xm.shape[0]
        z_date = self.conv_time(time_x.permute(0,2,1)).unsqueeze(-1)
        z_weather = self.conv_weather(weather_x.permute(0,2,1)).unsqueeze(-1)
        emb_xm = self.conv_xm(xm.permute(0,3,2,1))
        emb_xn = self.conv_xn(xn.permute(0,3,2,1))

        z_date_all_xm = z_date.repeat((1, 1, 1, self.num_xm)).permute(0, 1, 3, 2)
        z_weather_all_xm = z_weather.repeat((1, 1, 1, self.num_xm)).permute(0, 1, 3, 2)
        z_date_all_xn = z_date.repeat((1, 1, 1, self.num_xn)).permute(0, 1, 3, 2)
        z_weather_all_xn = z_weather.repeat((1, 1, 1, self.num_xn)).permute(0, 1, 3, 2)
        
        emb_xm_date = self.emb_layer_mode1_add_date(torch.cat((emb_xm, z_date_all_xm), dim=1).permute(0,2,1,3).flatten(2,3))
        emb_xm_weather = self.emb_layer_mode1_add_weather(torch.cat((emb_xm, z_weather_all_xm), dim=1).permute(0,2,1,3).flatten(2,3))
        emb_xn_date = self.emb_layer_mode2_add_date(torch.cat((emb_xn, z_date_all_xn), dim=1).permute(0,2,1,3).flatten(2,3))
        emb_xn_weather = self.emb_layer_mode2_add_weather(torch.cat((emb_xn, z_weather_all_xn), dim=1).permute(0,2,1,3).flatten(2,3))

        m2n_sim_mx_date = graph_attention(emb_xm_date, self.m2n_date_trans_params, emb_xn_date)
        m2n_sim_mx_weather = graph_attention(emb_xm_weather, self.m2n_weather_trans_params, emb_xn_weather)

        n2m_sim_mx_date = graph_attention(emb_xm_date, self.n2m_date_trans_params, emb_xn_date)
        n2m_sim_mx_weather = graph_attention(emb_xm_weather, self.n2m_weather_trans_params, emb_xn_weather)

        I_m2n = torch.zeros_like(self.predefined_adj).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.predefined_adj.device)
        I_n2m = torch.zeros_like(self.predefined_adj).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.predefined_adj.device)


        for i in range(self.num_relations):
            I_m2n_time = graph_topology(m2n_sim_mx_date, self.predefined_adj[i], 2, m2n_drop_percent)
            I_m2n_weather = graph_topology(m2n_sim_mx_weather, self.predefined_adj[i], 2, m2n_drop_percent)

            I_n2m_time = graph_topology(n2m_sim_mx_date, self.predefined_adj[i], 1, n2m_drop_percent)
            I_n2m_weather = graph_topology(n2m_sim_mx_weather, self.predefined_adj[i], 1, n2m_drop_percent)

            I_m2n[:, i, :, :] = (I_m2n_time + I_m2n_weather) / 2
            I_n2m[:, i, :, :] = (I_n2m_time + I_n2m_weather) / 2

        return I_m2n, I_n2m


class GatedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedFusion, self).__init__()
        # Define the gating unit, here using a convolutional layer as an example
        self.fc1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.activation = nn.ReLU()

    def forward(self, x1, x2):
        # x1 and x2 are the two feature maps to be fused
        # Apply the gating unit
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)
        Z_inter = self.fc1(x1)
        Z_intra = self.fc2(x2)
        
        # Concatenate and pass through sigmoid function
        gate_m = torch.sigmoid(torch.add(Z_inter, Z_intra))
        fused_feature = torch.mul(gate_m, x1) + torch.mul((1 - gate_m), x2)

        Z = self.fc3(fused_feature)
        out = self.activation(Z)
        out = self.fc4(out).permute(0,2,3,1)
        
        return out 

class SpatialConv(nn.Module):
    def __init__(self, seq_len, s_in_dim, s_out_dim):
        super(SpatialConv, self).__init__()
        self.spa_conv = HetGraphConv(seq_len, s_in_dim, s_out_dim)

    def forward(self, x, adj_mat):
        """
        :param mode1_x: (B, in_dim, T, N)
        :param t2t_mat: (N, N) mode1 to mode1
        :param r2t_mat: (N, N) mode2 to mode1
        :param mode2_x: (B, in_dim, T, N)
        :param r2r_mat: (N, N) mode2 to mode2
        :param t2r_mat: (N, N) mode1 to mode2
        :return:
        """
        out = self.spa_conv(hom_x=x, hom_mat=adj_mat)
        return out


class HetGraphConv(nn.Module):
    def __init__(self, cur_len, s_in_dim, s_out_dim):
        super(HetGraphConv, self).__init__()
        self.cur_len = cur_len
        self.s_in_dim = s_in_dim
        self.s_out_dim = s_out_dim
        
        self.W = nn.Conv2d(in_channels=2 * self.s_in_dim, out_channels=2 * self.s_out_dim, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1), bias=True)

        # self.drop_out = nn.Dropout(p=0.3)

    def forward(self, hom_x, hom_mat):
        """
        :param hom_x: (batch, in_dim, cur_len, num_nodes)
        :param hom_mat: (num_nodes, num_nodes)
        :param het_x: (batch, in_dim, cur_len, num_nodes)
        :param het_mat: (num_nodes, num_nodes)
        :return: (batch, out_dim, cur_len, num_nodes)
        """

        hom_conv = torch.einsum('bcln, nm -> bclm', (hom_x, hom_mat))

        out = torch.cat([hom_x, hom_conv], dim=1)

        out = self.W(out)  # (batch_size, 2 * out_dim, seq_len, num_nodes)
        out_p, out_q = torch.chunk(out, chunks=2, dim=1)
        out = out_p * torch.sigmoid(out_q)  # (batch_size, out_dim, seq_len, num_nodes)
        return out


class TemporalConv(nn.Module):
    def __init__(self, t_in_dim, t_out_dim, Kt, dilation, Ks=1):
        super(TemporalConv, self).__init__()
        self.Kt = Kt
        self.dilation = dilation

        self.conv2d = nn.Conv2d(t_in_dim, 2 * t_out_dim, kernel_size=(Kt, Ks),
                                     padding=((Kt - 1) * dilation, 0), dilation=dilation)

    def forward(self, x):
        """
        :param mode1_x: (batch, t_in_dim, seq_len, num_nodes)
        :param mode2_x: (batch, t_in_dim, seq_len, num_nodes)
        """
        
        x = self.conv2d(x)
        x = x[:, :, (self.Kt - 1) * self.dilation:-(self.Kt - 1) * self.dilation, :]

        x_p, x_q = torch.chunk(x, 2, dim=1)
        return x_p * torch.sigmoid(x_q)

class STBlock(nn.Module):
    def __init__(self, in_dim, tem_dim, spa_dim, Kt, dilation, cur_len, spatial_first):
        super(STBlock, self).__init__()
        self.spatial_first = spatial_first
        if self.spatial_first:
            self.spa_conv = SpatialConv(seq_len=cur_len,
                                             s_in_dim=in_dim, s_out_dim=spa_dim)
            self.tem_conv = TemporalConv(t_in_dim=spa_dim,
                                              t_out_dim=tem_dim, Kt=Kt, dilation=dilation)
            self.align_conv = ResAlign(in_dim, tem_dim)
            self.batch_norm = nn.BatchNorm2d(tem_dim)
        else:
            self.tem_conv = TemporalConv( t_in_dim=in_dim, t_out_dim=tem_dim, Kt=Kt, dilation=dilation)
            self.spa_conv = SpatialConv(seq_len=cur_len - (Kt - 1) * dilation,
                                             s_in_dim=tem_dim, s_out_dim=spa_dim)
            self.align_conv = ResAlign(in_dim, spa_dim)
            self.batch_norm = nn.BatchNorm2d(spa_dim)

        self.align_len = cur_len - (Kt - 1) * dilation
        
    def forward(self, x, adj_mat):
        """
        :param mode1_x: (B, in_dim, T, N)
        :param t2t_mat: (N, N) mode1 to mode1
        :param r2t_mat: (N, N) mode2 to mode1
        :param mode2_x: (B, in_dim, T, N)
        :param r2r_mat: (N, N) mode2 to mode2
        :param t2r_mat: (N, N) mode1 to mode2
        :return:
        """
        x_shortcut = self.align_conv(x[:, :, -self.align_len:, :])
        if self.spatial_first:
            x = self.spa_conv(x, adj_mat)
            x = self.tem_conv(x)
        else:
            x = self.tem_conv(x)
            x = self.spa_conv(x, adj_mat)
        x = self.batch_norm(x_shortcut + x)
        return x 

class ResAlign(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResAlign, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduce_conv = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))

    def forward(self, x):
        """
        Align the feature dimension
        :param x: (batch, in_dim, seq_len-(Kt-1), num_nodes)
        :return: (batch, out_dim, seq_len-(Kt-1), num_nodes)
        """
        if self.in_dim > self.out_dim:
            x = self.reduce_conv(x)
        elif self.in_dim < self.out_dim:
            batch, _, seq_len, num_nodes = x.shape
            x = torch.cat([x, torch.zeros([batch, self.out_dim - self.in_dim, seq_len, num_nodes],
                                          device=x.device)], dim=1)
        else:
            x = x
        return x 


def cal_adaptive_matrix(emb_s, emb_d):
    """
    :param emb_s: (num_nodes, emb_dim)
    :param emb_d: (num_nodes, emb_dim)
    :return:
    """
    dot_prod = torch.mm(emb_s, emb_d.t())
    adaptive_matrix = torch.softmax(torch.relu(dot_prod), dim=1)
    return adaptive_matrix
