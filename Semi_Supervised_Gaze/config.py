import os
# =============================================================================
# GazeFollow dataset dir config
# =============================================================================
gazefollow_base_dir = "/nfs/bigrod/add_disk0/qiaomu/datasets/gaze/gazefollow"
gazefollow_train_data = gazefollow_base_dir
gazefollow_train_label = os.path.join(gazefollow_base_dir, "train_annotations_release.txt")
gazefollow_val_data = gazefollow_base_dir
gazefollow_val_label = os.path.join(gazefollow_base_dir,"test_annotations_release.txt")


# =============================================================================
# VideoAttTarget dataset dir config
# =============================================================================
vat_base_dir = "/nfs/bigrod/add_disk0/qiaomu/datasets/gaze/videoattentiontarget"
videoattentiontarget_train_data = os.path.join(vat_base_dir, "images")
videoattentiontarget_train_label = os.path.join(vat_base_dir,"annotations/train")
videoattentiontarget_val_data = os.path.join(vat_base_dir, "images")
videoattentiontarget_val_label = os.path.join(vat_base_dir,"annotations/test")


# =============================================================================
# model config
# =============================================================================
input_resolution = 224
output_resolution = 64
