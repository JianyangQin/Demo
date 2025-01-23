import torch
import torch.nn as nn

'''
class calculate_edge_deletion_precent(nn.Module):
    """Complex weighted network to calculate edge deletion probability based on weather and environment."""
    def __init__(self, input_dim):
        super(calculate_edge_deletion_precent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)   # First layer   
        self.fc2 = nn.Linear(16, 8) # Second layer
        self.fc3 = nn.Linear(8, 1)            # Output layer
        #self.dropout = nn.Dropout(0.2)          # Dropout layer to prevent overfitting
        self.relu = nn.ReLU()                    # ReLU activation function

    def forward(self, time_vector, weather_vector):
        context = torch.cat((time_vector.squeeze(), weather_vector.squeeze()), dim=1)  # Correct concatenation
        context = self.relu(self.fc1(context))              # First layer
        context = self.relu(self.fc2(context))              # Second layer
        drop_percent = torch.sigmoid(self.fc3(context))             # Third layer
        #print(drop_percent)
        #if torch.isnan(drop_percent).any():
        #    print( time_vector)
        #    print(weather_vector)
        return drop_percent      # Output layer, sigmoid ensures values are in [0, 1]
'''

def graph_attention(q, k, v):
    qk = torch.einsum('bmc, nc -> bmn', q, k)
    return torch.einsum('bmc, bnc -> bmn', qk, v)

# def graph_topology(sim_mx, input_graph, softmax_dim, drop_precent):
#     """Generate data augmentation from topology (graph structure) perspective for undirected graph without self-loop."""
#
#     ## Edge dropping starts here
#     batch_size = sim_mx.shape[0]
#     input_graph = input_graph.repeat(batch_size, 1, 1)
#
#     # sim_mx[input_graph == 0.] = 0.
#
#     del_mx = sim_mx
#     add_mx = sim_mx.clone()
#
#     # 创建一个布尔掩码，标记零值位置
#     zero_mask = (input_graph == 0).float()  # 转换为float类型，以便后续操作
#
#     # 将零值替换为一个非常小的负数（避免影响softmax的计算），并记录这些位置
#     # 注意：这里选择-1e9是因为它是一个相对较小的数，对softmax的计算结果影响可以忽略不计
#     # 但你需要确保这个数足够小，不会改变softmax的主要结果
#     del_mx[zero_mask.bool()] = -1e9
#     add_mx[(1-zero_mask).bool()] = -1e9
#
#     # 对每一行应用softmax
#     del_prob = torch.softmax(del_mx, dim=softmax_dim)
#     add_prob = torch.softmax(add_mx, dim=softmax_dim)
#
#     del_mask = (del_prob < 0.3).float()
#     add_mask = (add_prob > 0.7).float()
#
#     # 使用掩码恢复零值位置
#     prob_del = del_prob * (1 - zero_mask) * (1 - del_mask)
#     prob_add = add_prob * zero_mask * add_mask
#
#     prob_final = prob_del + prob_add
#     prob_final = torch.softmax(prob_final, dim=softmax_dim)
#     prob_final = prob_final * ((1 - del_mask) + add_mask)
#
#     return prob_final

def graph_topology(sim_mx, input_graph, softmax_dim, drop_precent):
    """Generate data augmentation from topology (graph structure) perspective for undirected graph without self-loop."""

    ## Edge dropping starts here
    batch_size = sim_mx.shape[0]
    input_graph = input_graph.repeat(batch_size, 1, 1)

    sim_mx[input_graph == 0.] = 0.

    # 创建一个布尔掩码，标记零值位置
    zero_mask = (sim_mx == 0).float()  # 转换为float类型，以便后续操作

    # 将零值替换为一个非常小的负数（避免影响softmax的计算），并记录这些位置
    # 注意：这里选择-1e9是因为它是一个相对较小的数，对softmax的计算结果影响可以忽略不计
    # 但你需要确保这个数足够小，不会改变softmax的主要结果
    sim_mx[sim_mx == 0] = -1e9

    # 对每一行应用softmax
    prob = torch.softmax(sim_mx, dim=softmax_dim)
    drop_mask = (prob < drop_precent).float()

    # 使用掩码恢复零值位置
    prob = prob * (1 - zero_mask) * (1 - drop_mask)
    # 或者更简洁地：
    prob[zero_mask.bool()] = 0

    return prob



def cf_topology(sim_mx, input_graph, drop_precent):
    """Generate data augmentation from topology (graph structure) perspective for undirected graph without self-loop."""
 
    ## Edge dropping starts here
    batch_size = sim_mx.shape[0]
    input_graph = input_graph.repeat(batch_size,  1, 1)

    #print(input_graph.shape)
    index_list = input_graph.nonzero()  # list of edges [row_idx, col_idx]
    #print(index_list)
    edge_num = len(index_list)  
    edge_mask = (input_graph > 0)
    #print(input_graph.shape)
    #print(edge_mask)
    #print(sim_mx)
    #print((sim_mx[edge_mask]))
    #print(edge_num.shape)
    #print(drop_precent.shape) # Use .item() to get a scalar
    #print(add_drop_num)
    #print(sim_mx.shape)
    mask = torch.zeros_like(input_graph, dtype=torch.int64).to(input_graph.device)
    '''
    for i in range(batch_size):
        #print(drop_precent[i])
        #print(edge_num * drop_precent[i])
        add_drop_num = int(edge_num * drop_precent) 
        if add_drop_num == 0 :
            mask[i] = 0
        else:
            drop_prob = torch.softmax(sim_mx[i, edge_mask], dim = 0)
            #print(drop_prob.shape)
            drop_prob = (1. - drop_prob)  # normalized similarity to get sampling probability 
            drop_prob /= drop_prob.sum()
            
            #print(drop_prob.shape)

            drop_list_indices = torch.multinomial(drop_prob, add_drop_num, replacement=False)  # Sample indices based on probability
            drop_index = index_list[drop_list_indices]
            #print(drop_index)
            #print(drop_index)
            # print(mask.device)
            # Set selected indices in the mask to 0
            mask[i, drop_index[:,0], drop_index[:,1]] = 1
            
            for j in drop_list_indices:
                mask[i, index_list[j][0], index_list[j][1]] = 1
            '''
    add_drop_num = int(edge_num * drop_precent) 
    drop_prob = torch.softmax(sim_mx[edge_mask], dim = 0)
    #print(drop_prob.shape)
    drop_prob = (1. - drop_prob)  # normalized similarity to get sampling probability 
    drop_prob /= drop_prob.sum()
    
    #print(drop_prob.shape)

    drop_list_indices = torch.multinomial(drop_prob, add_drop_num, replacement=False)  # Sample indices based on probability
    #print(drop_list_indices.shape)
    drop_index = index_list[drop_list_indices]
    #print(drop_index)
    #print(drop_index)
    # print(mask.device)
    # Set selected indices in the mask to 0
    mask[drop_index[:,0],drop_index[:,1], drop_index[:,2]] = 1
    
    #for j in drop_list_indices:
    #    mask[i, index_list[j][0], index_list[j][1]] = 1
      # Create the augmented graph by applying the mask
    #print(mask.dtype)
    return mask