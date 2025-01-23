import torch
import torch.nn as nn 

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
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        c1 = self.fc1(x1)
        c2 = self.fc2(x2)

        # Concatenate and pass through sigmoid function
        gate_m = torch.sigmoid(torch.add(c1, c2))
        fused_feature = torch.mul(gate_m, x1) + torch.mul((1 - gate_m), x2)

        out = self.fc3(fused_feature)
        out = self.activation(out)
        out = self.fc4(out).permute(0, 2, 3, 1)

        return out


class SpatialConv(nn.Module):
    def __init__(self, seq_len, s_in_dim, s_out_dim):
        super(SpatialConv, self).__init__()
        self.spa_conv = HetGraphConv(seq_len, s_in_dim, s_out_dim)

    def forward(self, x, adj_mat):
        out = self.spa_conv(het_x=x, het_mat=adj_mat)
        return out


class HetGraphConv(nn.Module):
    def __init__(self, cur_len, s_in_dim, s_out_dim):
        super(HetGraphConv, self).__init__()
        self.cur_len = cur_len
        self.s_in_dim = s_in_dim
        self.s_out_dim = s_out_dim

        self.W = nn.Conv2d(
            in_channels=2 * self.s_in_dim, 
            out_channels=2 * self.s_out_dim, 
            kernel_size=(1, 1),
            padding=(0, 0), 
            stride=(1, 1), 
            bias=True
        )


    def forward(self, het_x, het_mat):
        """
        :param het_x: (batch, in_dim, cur_len, num_nodes)
        :param het_mat: (num_nodes, num_nodes)
        :return: (batch, out_dim, cur_len, num_nodes)
        """

        hom_conv = torch.einsum('bcln, nm -> bclm', (het_x, het_mat))

        out = torch.cat([het_x, hom_conv], dim=1)

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
        :param x: (batch, t_in_dim, seq_len, num_nodes)
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
            self.spa_conv = SpatialConv(seq_len=cur_len, s_in_dim=in_dim, s_out_dim=spa_dim)
            self.tem_conv = TemporalConv(t_in_dim=spa_dim, t_out_dim=tem_dim, Kt=Kt, dilation=dilation)
            self.align_conv = ResAlign(in_dim, tem_dim)
            self.batch_norm = nn.BatchNorm2d(tem_dim)
        else:
            self.tem_conv = TemporalConv(t_in_dim=in_dim, t_out_dim=tem_dim, Kt=Kt, dilation=dilation)
            self.spa_conv = SpatialConv(seq_len=cur_len - (Kt - 1) * dilation, s_in_dim=tem_dim, s_out_dim=spa_dim)
            self.align_conv = ResAlign(in_dim, spa_dim)
            self.batch_norm = nn.BatchNorm2d(spa_dim)

        self.align_len = cur_len - (Kt - 1) * dilation

    def forward(self, x, adj_mat):
        """
        :param x: (B, in_dim, T, N)
        :param adj_mat: (N, N) mode1 to mode1
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