#import torch


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def OPML_config():
    config = Config({
        # model setting
        'num_vps': 8,                       # number of viewports in a sequence.
        'img_channels': 3,
        'img_size': 224,
        'dim': 64,                          # dimension after Stem module.
        'depths': (2, 2, 5, 3),             # number of maxvit block in each stage.
        'channels': (128, 256, 512, 512),     # channels in each stage.
        'num_heads': (2, 4, 8, 16),          # number of head in each stage.
        
        'mlp_ratio': 3,
        'drop_rate': 0.,
        'pos_drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.,              # droppath rate in encoder block.
        'kernel_size': 7,
        'layer_scale': None,
        'dilations': None,
        'qkv_bias': True,
        'qk_scale': None,
        'select_rate': 0.5,                 # the rate of select feature from all viewport features.
        'num_classes': 4,
        'hidden_dim': 1152,                   
        
        
        # resource setting
        'root_dir':'/mnt/10T/zjy/database/OIQ_10k/OIQ_10k_resize_512',
        'csv_path':'/mnt/10T/zjy/D_OIQA/database_csv/RMCP/OIQ-10k/GPT_4v_4o_ge_cl_OIQ_10k_train.csv',
        'test_csv_path':'/mnt/10T/zjy/D_OIQA/database_csv/RMCP/OIQ-10k/GPT_4v_4o_ge_cl_OIQ_10k_test.csv',
        'Long-CLIP_path':'XXX/XXX/XXX'
        #'vp_path': '/mnt/10T/tzw/methods/dataset/OIQ-10K/viewports_8',
        # 'vp_path':'/mnt/10T/rjl/dataset/viewports_8',
        #'train_info_csv_path': '/mnt/10T/zjy/D_OIQA/database_csv/mutil_csv/GPT_4_JUFE_10k_train.csv',
        #'test_info_csv_path': '/mnt/10T/zjy/D_OIQA/database_csv/mutil_csv/GPT_4_JUFE_10k_test.csv',
        #'save_ckpt_path': '/home/test/10t/tzw/ckpt/IQCaption360',
        'load_ckpt_path': '',
        'tensorboard_path': '',
        # train setting
        'seed': 42,
        #'model_name': 'IQCaption360-OIQ10K-AFA-MSFS-VPFS-DRPN-QSPN-lr1e-4-bs32-epoch50',
        #'dataset_name': 'OIQ-10K_GPT_4_vision_preview',
        'epochs': 50,
        'batch_size': 8,
        'num_workers': 8,
        'learning_rate': 1e-4,
        'lrf': 0.01,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'p': 1,
        'q': 2,
        'use_tqdm': False,
        'T_max':10,
        'eta_min':0,
        'use_tensorboard': False,
        'batch_print': False,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'val_freq':1,
        
         # load & save checkpoint
        "model_name": "Cross-batch_8_test1",
        "type_name": "OIQ_10k_ablition",
        "ckpt_path": "/mnt/10T/zjy/D_OIQA/RMCP_model/output/models/RMCP",  # directory for saving checkpoint
        "log_path": "/mnt/10T/zjy/D_OIQA/RMCP_model/output/log/RMCP",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/"
    })  
        
    return config
