import torch
import torch.nn as nn


class Causal_FlowTrans(nn.Module):
    def __init__(self, predefined_adj, hidden_dim, num_of_xm, num_of_xn, len_input_xm, len_input_xn,
                 input_dim, time_dim, weather_dim, num_relations, scaling):
        super(Causal_FlowTrans, self).__init__()
        self.scaling = scaling
        self.predefined_adj = predefined_adj
        self.causal_trans_prob = Causal_TransProb(
            predefined_adj, hidden_dim, num_of_xm, num_of_xn,
            len_input_xm, len_input_xn, input_dim, time_dim, weather_dim
        )
        self.mgcn_multimodal = MultiGraphConv(num_relations)

    def forward(self, xm, xn, f_mo, f_no, time_x, weather_x,
                m2n_drop_percent=0.6, n2m_drop_percent=0.6):
        """
        Forward pass for dynamic causal interaction.

        Args:
            x_m (torch.Tensor): Input tensor for mode m, shape (batch_size, time_steps, node_count_mode_m, hidden_dim).
            x_n (torch.Tensor): Input tensor for mode n, shape (batch_size, time_steps, node_count_mode_n, hidden_dim).
            f_mo (torch.Tensor): Inter-modal feature for mode m, shape (batch_size, time_steps, node_count_mode_m, hidden_dim).
            f_no (torch.Tensor): Inter-modal feature for mode n, shape (batch_size, time_steps, node_count_mode_n, hidden_dim).
            time_x (torch.Tensor): Input tensor representing date features.
            weather_x (torch.Tensor): Input tensor representing weather features.
            m2n_drop_percent: (torch.float): Mask the low probability values in Transition Graph from mode m to n
            n2m_drop_percent: (torch.float): Mask the low probability values in Transition Graph from mode n to m

        Returns:
            torch.Tensor: Generated interactions.
        """
        mask = self.predefined_adj.unsqueeze(0).repeat(xm.shape[0], 1, 1, 1)

        I_m2n, I_n2m = self.causal_trans_prob(
            xm, xn, time_x, weather_x, m2n_drop_percent, n2m_drop_percent
        )

        _I_m2n = torch.div(mask * I_m2n, self.scaling)
        _I_n2m = torch.div(mask * I_n2m, self.scaling)

        Z_m2n, Z_n2m = self.mgcn_multimodal(f_mo, f_no, _I_m2n, _I_n2m)

        Z_m2n = Z_m2n + f_no
        Z_n2m = Z_n2m + f_mo

        return Z_n2m, Z_m2n, I_n2m, I_m2n


class MultiGraphConv(nn.Module):
    def __init__(self, num_relations):
        """
        Multi-Relation Graph Convolutional Layer to model cross-modal interactions with multiple relations.

        Args:
            num_relations (int): Number of different relations between mode m and mode n nodes.
        """
        
        super(MultiGraphConv, self).__init__()

        # Learnable weights for combining different relations
        self.relation_weights_mode1tmode2 = nn.Parameter(torch.FloatTensor(num_relations))
        nn.init.constant_(self.relation_weights_mode1tmode2, 1.0 / num_relations)  # Initially uniform weights
        self.relation_weights_mode2tmode1 = nn.Parameter(torch.FloatTensor(num_relations))
        nn.init.constant_(self.relation_weights_mode2tmode1, 1.0 / num_relations)  # Initially uniform weights

    def forward(self, x_m, x_n, I_m2n, I_n2m):
        """
        Forward pass for multi-relation interaction graph convolution.

        Args:
            x_m (torch.Tensor): Input tensor for mode m, shape (batch_size, time_steps, node_count_mode_m, hidden_dim).
            x_n (torch.Tensor): Input tensor for mode n, shape (batch_size, time_steps, node_count_mode_n, hidden_dim).
            interaction_adj (torch.Tensor): Multi-relation interaction adjacency matrix, shape (num_relations, node_count_mode1, node_count_mode2).

        Returns:
            torch.Tensor: Combined interaction output across multiple relations.
        """
        
        num_relations = I_m2n.size(1)

        # Collect outputs for each relation
        relation_outputs_n2m = []
        relation_outputs_m2n = []
        for r in range(num_relations):
            output_n2m = torch.einsum('blnc, bmn -> blmc', x_n, I_n2m[:, r, :, :].to(torch.float32))
            output_m2n = torch.einsum('blmc, bnm -> blnc', x_m, I_m2n[:, r, :, :].transpose(1, 2).to(torch.float32))

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


class Causal_TransProb(nn.Module):
    def __init__(self, predefined_adj, hidden_dim, num_of_xm, num_of_xn, len_input_xm, len_input_xn, input_dim,
                 time_dim, weather_dim):
        super(Causal_TransProb, self).__init__()
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
        self.conv_time = nn.Conv1d(in_channels=2, out_channels=self.hidden_dim // 4, kernel_size=time_dim)
        self.conv_weather = nn.Conv1d(in_channels=2, out_channels=self.hidden_dim // 4, kernel_size=weather_dim)

        self.conv_xm = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim // 4, kernel_size=1)
        self.conv_xn = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim // 4, kernel_size=1)

        self.m2n_date_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.m2n_weather_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.n2m_date_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.n2m_weather_trans_params = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.emb_layer_mode1_add_date = nn.Linear(self.hidden_dim // 2 * self.input_len_xm, self.embed_dim)
        self.emb_layer_mode2_add_date = nn.Linear(self.hidden_dim // 2 * self.input_len_xn, self.embed_dim)
        self.emb_layer_mode1_add_weather = nn.Linear(self.hidden_dim // 2 * self.input_len_xm, self.embed_dim)
        self.emb_layer_mode2_add_weather = nn.Linear(self.hidden_dim // 2 * self.input_len_xn, self.embed_dim)

    def forward(self, xm, xn, time_x, weather_x, n2m_drop_percent=0.6, m2n_drop_percent=0.6):
        """
        Forward pass for multi-relation interaction graph convolution.

        Args:
            x_m (torch.Tensor): Input tensor for mode m, shape (batch_size, time_steps, node_count_mode_m, hidden_dim).
            x_n (torch.Tensor): Input tensor for mode n, shape (batch_size, time_steps, node_count_mode_n, hidden_dim).
            time_x (torch.Tensor): Time information, shape (batch_size, time_steps, dims).
            weather_x (torch.Tensor): Weather information, shape (batch_size, time_steps, dims).

        Returns:
            torch.Tensor: Transition Probability Graph
        """

        batch_size = xm.shape[0]

        # embedding of traffic flow, time, and weather
        z_date = self.conv_time(time_x.permute(0, 2, 1)).unsqueeze(-1)
        z_weather = self.conv_weather(weather_x.permute(0, 2, 1)).unsqueeze(-1)
        emb_xm = self.conv_xm(xm.permute(0, 3, 2, 1))
        emb_xn = self.conv_xn(xn.permute(0, 3, 2, 1))

        z_date_all_xm = z_date.repeat((1, 1, 1, self.num_xm)).permute(0, 1, 3, 2)
        z_weather_all_xm = z_weather.repeat((1, 1, 1, self.num_xm)).permute(0, 1, 3, 2)
        z_date_all_xn = z_date.repeat((1, 1, 1, self.num_xn)).permute(0, 1, 3, 2)
        z_weather_all_xn = z_weather.repeat((1, 1, 1, self.num_xn)).permute(0, 1, 3, 2)

        # jointly encode traffic flow, time, and weather
        emb_xm_date = self.emb_layer_mode1_add_date(
            torch.cat((emb_xm, z_date_all_xm), dim=1).permute(0, 2, 1, 3).flatten(2, 3))
        emb_xm_weather = self.emb_layer_mode1_add_weather(
            torch.cat((emb_xm, z_weather_all_xm), dim=1).permute(0, 2, 1, 3).flatten(2, 3))
        emb_xn_date = self.emb_layer_mode2_add_date(
            torch.cat((emb_xn, z_date_all_xn), dim=1).permute(0, 2, 1, 3).flatten(2, 3))
        emb_xn_weather = self.emb_layer_mode2_add_weather(
            torch.cat((emb_xn, z_weather_all_xn), dim=1).permute(0, 2, 1, 3).flatten(2, 3))

        # calculate probability graph
        m2n_sim_mx_date = calculate_probability_graph(emb_xm_date, self.m2n_date_trans_params, emb_xn_date)
        m2n_sim_mx_weather = calculate_probability_graph(emb_xm_weather, self.m2n_weather_trans_params, emb_xn_weather)

        n2m_sim_mx_date = calculate_probability_graph(emb_xm_date, self.n2m_date_trans_params, emb_xn_date)
        n2m_sim_mx_weather = calculate_probability_graph(emb_xm_weather, self.n2m_weather_trans_params, emb_xn_weather)

        # mask probability graph to focus on key stations and high probabilities
        I_m2n = torch.zeros_like(self.predefined_adj).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(
            self.predefined_adj.device)
        I_n2m = torch.zeros_like(self.predefined_adj).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(
            self.predefined_adj.device)

        for i in range(self.num_relations):
            I_m2n_time = mask_probability_graph(m2n_sim_mx_date, self.predefined_adj[i], 2, m2n_drop_percent)
            I_m2n_weather = mask_probability_graph(m2n_sim_mx_weather, self.predefined_adj[i], 2, m2n_drop_percent)

            I_n2m_time = mask_probability_graph(n2m_sim_mx_date, self.predefined_adj[i], 1, n2m_drop_percent)
            I_n2m_weather = mask_probability_graph(n2m_sim_mx_weather, self.predefined_adj[i], 1, n2m_drop_percent)

            I_m2n[:, i, :, :] = (I_m2n_time + I_m2n_weather) / 2
            I_n2m[:, i, :, :] = (I_n2m_time + I_n2m_weather) / 2

        return I_m2n, I_n2m
    
def calculate_probability_graph(q, k, v):
    qk = torch.einsum('bmc, nc -> bmn', q, k)
    return torch.einsum('bmc, bnc -> bmn', qk, v)
    
def mask_probability_graph(sim_mx, input_graph, softmax_dim, drop_precent):
    """
    Generate dynamic transition probability by masking
    """

    batch_size = sim_mx.shape[0]
    input_graph = input_graph.repeat(batch_size, 1, 1)

    # mask irrelevant relationships for softmax
    sim_mx[input_graph == 0.] = 0.
    sim_mx[sim_mx == 0] = -1e9

    # obtain the masks of irrelevant relationships
    zero_mask = (sim_mx == 0).float()

    # softmax to obtain transition probability
    prob = torch.softmax(sim_mx, dim=softmax_dim)

    # obtain the masks of low transition probability
    drop_mask = (prob < drop_precent).float()

    # obtain the final transition probability
    prob = prob * (1 - zero_mask) * (1 - drop_mask)

    return prob