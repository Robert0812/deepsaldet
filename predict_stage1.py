# imports
import os
import sys
import numpy as np
from glob import glob
import pylab as pl
from skimage.segmentation import slic
from skimage.util import pad
from skimage.io import imsave
import leveldb
import datum
import argparse
import cPickle
from multiprocessing import Pool

# image padding for testing
def image_padding(img, padding_value=0, crop_size=None, ismask=0):
    h, w = img.shape[:2]
    if crop_size is None:
        pad_h = h
        pad_w = w
    else:
        if h > w:
            pad_h = int(crop_size/2.)
            pad_w = int(1.*pad_h*w/h)
        else:
            pad_w = int(crop_size/2.)
            pad_h = int(1.*pad_w*h/w)
    
    if not ismask:
        return pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=int(padding_value)), pad_h, pad_w
    else:
        return pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0), pad_h, pad_w

# convert leveldb to ndarray
def leveldb2ndarray(dbfolder, num_feat, dim_feat):
	db = leveldb.LevelDB(dbfolder)
	dt = datum.Datum()

	output = np.zeros((num_feat, dim_feat))
	for idx in range(num_feat):
	    dt.ParseFromString(db.Get('%d' %(idx)))
	    output[idx, :] = dt.float_data

	return output

def main():

	# setting parameters 
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--cafferoot', help='caffe root', required=True)
    	parser.add_argument('-w', '--weights', help='pretrained model weights', required=True)
	parser.add_argument('-m', '--model', help='model definition file', required=True)
	parser.add_argument('-b', '--blob', help='output feature blob', required=True)
	parser.add_argument('-t', '--testfolder', help='a folder of images for testing', required=True)
	parser.add_argument('-n', '--slic_n_segments', help='number of superpixels for testing', type=int, default=100)
	parser.add_argument('-c', '--slic_compactness', help='compactness in superpixel segmentation', type=int, default=10)
	parser.add_argument('-g', '--gpu', help='set GPU device id', type=int, default=0)
	args = parser.parse_args()

   	FEAT_TOOL = args.cafferoot + 'build/tools/extract_features.bin'

	# load test images
	IMG_EXT='.jpg'
	MSK_EXT='.png'
	img_files = sorted(glob(args.testfolder + '*' + IMG_EXT), key=lambda name: int(os.path.basename(name)[:-4]))
	print 'Loading images ...'
	imgs = [pl.imread(imf) for imf in img_files]
    	print 'Superpixel segmentation ...'
	segfunc = lambda im: slic(im, n_segments=args.slic_n_segments, compactness=args.slic_compactness, sigma=1)
	segs = map(segfunc, imgs)

	## unified steps ##
	# 1. generate padded images 
	# 2. apply superpixel segmentation to test images
	# 3. generate window files with a list of windows centering at superpixel centers
	if os.path.isdir('./_tmp'):
	    os.system('rm -rf ./_tmp')
	os.system('mkdir -p ./_tmp/imgs')
	os.system('mkdir -p ./_tmp/smaps_sc')

	f = open('./_tmp/window_file_test.txt', 'wb')

	window_cnt = []
	cropping_factor = 1.0
	for i in range(len(img_files)):
	    img = imgs[i]
	    # padding
	    pimg, pad_h, pad_w = image_padding(img, padding_value=114.452, crop_size=None)
	    pimg_file = './_tmp/imgs/'+os.path.basename(img_files[i])
	    imsave(pimg_file, pimg)
	    # SLIC
	    seg = segs[i]
	    # window list
	    ctr_xs = []
	    ctr_ys = []
	    pad_ws = int(pad_w*cropping_factor)
    	    pad_hs = int(pad_h*cropping_factor)
	    for sid in np.unique(seg):
	        # todo: augment data by scaling and rotation 
	        idx = seg == sid
	        ys, xs = np.where(idx)
	        ctr_xs.append(int(xs.mean())+pad_w)
	        ctr_ys.append(int(ys.mean())+pad_h)
	    windows = [[ctr_x-pad_ws, ctr_y-pad_hs, ctr_x+pad_ws, ctr_y+pad_hs] for ctr_x, ctr_y in zip(ctr_xs, ctr_ys)]
	    
	    # save windows into file
	    f.write('# {}\n'.format(i))
	    f.write(pimg_file)
	    f.write('\n{}\n{}\n{}\n'.format(3, pimg.shape[0], pimg.shape[1]))
	    f.write('{}\n'.format(len(windows)))
	    window_cnt.append(len(windows))
	    
	    for j in range(len(windows)):
	        x1 = windows[j][0]
	        y1 = windows[j][1]
	        x2 = windows[j][2]
	        y2 = windows[j][3]
	        f.write('{0:d} {1:.3f} {2:d} {3:d} {4:d} {5:d}\n'.format(0, 0, x1, y1, x2, y2))
	        
	    print 'Extracting windows from {}-th image ... [Total: {}]'.format(i, len(img_files))
	        
	f.close()

	num_window = sum(window_cnt)
	num_batch = int(num_window/16)+1
	print num_window

	leveldb_folder = './_tmp/feats'
	if os.path.isdir(leveldb_folder):
	    os.system('rm -rf '+ leveldb_folder)
	# print 'Extracting features for {} input images ...'.format(num_window)
	if args.gpu is None:
		os.system('{} {} {} {} {} {}'.format(FEAT_TOOL, args.weights, args.model, args.blob, leveldb_folder, num_batch))
	else:
		os.system('{} {} {} {} {} {} GPU {}'.format(FEAT_TOOL, args.weights, args.model, args.blob, leveldb_folder, num_batch, args.gpu))

	outputs = leveldb2ndarray(dbfolder=leveldb_folder, num_feat=num_window, dim_feat=2)

	labels = []
	out_idx = -1
	enhance = 0
	for i in range(len(img_files)):
	    seg = segs[i]
	    label = np.zeros_like(seg, dtype=float)
	    for sid in np.unique(seg):
	        idx = seg == sid
	        out_idx += 1
	        # perform softmax to output
	        label[idx] = np.exp(outputs[out_idx, 1])/np.exp(outputs[out_idx, :]).sum()
	    if enhance:
	    	tmp = np.exp(1.25 * label)
    	    else:
	    	tmp = label
    	    tmp = 1.0*(tmp-tmp.min())/(tmp.max()-tmp.min()+sys.float_info.epsilon)
	    imsave('./_tmp/smaps_sc/'+'{}_sc.png'.format(os.path.basename(img_files[i])[:-4]), tmp)
	    labels.append(label)

if __name__ == '__main__':

	main()
