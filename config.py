seq_len = 4
pred_len = 1

nyc_mb_config = {
    'CausalMTF': {
                  'num_nodes_xm': 426,
                  'num_nodes_xn': 226,
                  'pred_mat': "pre_mat_nyc_mb.npy",
                  'emb_dim': 20,
                  'input_dim': 1,
                  'time_dim': 1,
                  'weather_dim': 1,
                  'hidden_dim': 64,
                  'interval_xm': 60,
                  'interval_xn': 60,
                  'Kt': [2, 2],
                  'dilation': [1, 2],
                  'spatial_first': False,
                  'seq_len': 4 ,
                  'pred_len': 1 ,
                  'save_path':'nyc_mb'
                 }
}

nyc_mt_config = {
    'CausalMTF': {
                  'num_nodes_xm': 426,
                  'num_nodes_xn': 82,
                  'pred_mat': "pre_mat_nyc_mt.npy",
                  'emb_dim': 20,
                  'input_dim': 1,
                  'time_dim': 1,
                  'weather_dim': 1,
                  'hidden_dim': 64,
                  'interval_xm': 60,
                  'interval_xn': 60,
                  'Kt': [2, 2],
                  'dilation': [1, 2],
                  'spatial_first': False,
                  'seq_len': 4 ,
                  'pred_len': 1 ,
                  'save_path':'nyc_mt'
                 }
}

nyc_bt_config = {
    'CausalMTF': {
                  'num_nodes_xm': 226,
                  'num_nodes_xn': 82,
                  'pred_mat': "pre_mat_nyc_bt.npy",
                  'emb_dim': 20,
                  'input_dim': 1,
                  'time_dim': 1,
                  'weather_dim': 1,
                  'hidden_dim': 64,
                  'interval_xm': 60,
                  'interval_xn': 60,
                  'Kt': [2, 2],
                  'dilation': [1, 2],
                  'spatial_first': False,
                  'seq_len': 4 ,
                  'pred_len': 1 ,
                  'save_path':'nyc_bt'
                 }
}
