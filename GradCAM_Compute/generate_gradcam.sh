
#export PYTHONPATH=$PYTHONPATH:$PWD/fairseq
CUDA_VISIBLE_DEVICES=0 python gradcam_gen_largescale.py --base_dir /nfs/bigrod/add_disk0/qiaomu/datasets/gaze/gazefollow --sample_num 50 --vis_gradcam --save_folder test_gradcam