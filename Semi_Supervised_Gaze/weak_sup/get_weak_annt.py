import os
import random
import numpy as np
import pandas as pd


data_dir = "/data/add_disk0/qiaomu/datasets/gaze/gazefollow"
save_dir = os.path.join(data_dir, 'weak_supervision')
train_annt = os.path.join(data_dir, "train_annotations_release_persondet.txt")
column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta', 'ori_name',
                            'body_x1', 'body_y1', 'body_x2', 'body_y2']

ratio = 0.05

df = pd.read_csv(train_annt, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
df=df[df['inout']!=-1]
df = df[np.logical_and(np.less_equal(df['bbox_x_min'].values,df['bbox_x_max'].values), np.less_equal(df['bbox_y_min'].values, df['bbox_y_max'].values))]
df.reset_index(inplace=True, drop=True)

num_samples = df.shape[0]
num_select = round(num_samples * ratio)
rand_idx = np.random.permutation(np.arange(num_samples))
split1_idx, split2_idx = rand_idx[:num_select].tolist(), rand_idx[num_select:].tolist()

df_split_1, df_split_2 = df.iloc[split1_idx,:].copy(deep=True), df.iloc[split2_idx,:].copy(deep=True)
#df_split_1.reset_index(inplace=True)
#df_split_2.reset_index(inplace=True)
    
df_split_1.to_csv(os.path.join(save_dir, f'train_annotations_weak_ratio{ratio}_use.txt'), header=False, index=False) # annt used for semi-supervised training
df_split_2.to_csv(os.path.join(save_dir, f'train_annotations_weak_ratio{ratio}_left.txt'), header=False, index=False) # annt left out, which is not for training



