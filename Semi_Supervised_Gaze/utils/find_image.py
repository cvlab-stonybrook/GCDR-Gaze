import os
import numpy as np
import cv2
import pdb
from tqdm import tqdm

gazefollow_base_dir = "/data/add_disk0/qiaomu/datasets/gaze/gazefollow"
videoatt_base_dir = "/nfs/bigcortex/add_disk0/qiaomu/datasets/gaze/videoattentiontarget"

annt_path = os.path.join(gazefollow_base_dir, 'test_annotations_release.txt')

all_imgs = []
img_match = {}
#for imgfile in os.listdir('./test_pics'):
	#imgpath = os.path.join('./test_pics', imgfile)
	#img = cv2.imread(imgpath)
	#img = cv2.resize(img, (224,224))
	#img_match[imgfile] = [img, 255, '']

imgfile = os.path.join(gazefollow_base_dir, "visualizations", "select_diff.jpg")
print(os.path.basename(imgfile))
img_query = cv2.imread(imgfile)
img_query = cv2.resize(img_query, (224,224))
img_query_mean = img_query.reshape(-1,3).mean(axis=0)[None, None, :]
img_query_std = img_query.reshape(-1,3).std(axis=0)[None, None, :]
img_query = (img_query - img_query_mean) / img_query_std

all_imgs, all_img_dist = [], np.array([])
last_img = ''
img_set = set()
with open(annt_path, 'r') as file:
	lines = file.readlines()
	for line in tqdm(lines):
		this_path = line.split(',')[0]
		if this_path in img_set:
			continue
		img_set.add(this_path)
		imgpath = os.path.join(gazefollow_base_dir, this_path)
		if this_path==last_img:
			continue

		last_img = this_path
		img = cv2.imread(imgpath)
		img = cv2.resize(img, (224,224))
		img_mean = img.reshape(-1,3).mean(axis=0)[None, None, :]
		img_std = img.reshape(-1,3).std(axis=0)[None, None, :]
		img = (img - img_mean) / img_std
		if len(img.shape)<3:
			continue
		
		img_diff = np.abs(img - img_query).mean()
		if len(all_img_dist)<10:
			all_img_dist = np.append(all_img_dist, img_diff)
			all_imgs.append(this_path)
			continue
		if img_diff >= all_img_dist[-1]:
			continue
		all_img_dist = np.append(all_img_dist, img_diff)
		all_imgs.append(this_path)
		sort_idx = np.argsort(all_img_dist)[:10].astype(int)
		all_img_dist = all_img_dist[sort_idx]
		#pdb.set_trace()
		all_imgs = [all_imgs[i] for i in sort_idx]
		

pdb.set_trace()

