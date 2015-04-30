caffe_root=../caffe-sal
testfolder=./images
device_id=0

cd ${caffe_root}/data/ilsvrc12 && sh get_ilsvrc_aux.sh && cd -
mean_file="    mean_file:\"${caffe_root}/data/ilsvrc12/imagenet_mean.binaryproto\""
sed -e "9s@.*@${mean_file}@" clarifai_stage1.prototxt > tmpfile; mv tmpfile clarifai_stage1.prototxt
sed -e "10s@.*@${mean_file}@" clarifai_stage2.prototxt > tmpfile; mv tmpfile clarifai_stage2.prototxt

python predict_stage1.py \
	--cafferoot ${caffe_root}/ \
       	--weights ./models/model_imagenet_padimage_pretrain_iter_600000_stage1_iter_80000.caffemodel \
	--model ./clarifai_stage1.prototxt \
	--blob fc8_msra10k \
	--testfolder ${testfolder}/ \
	--slic_n_segments 100 \
	--slic_compactness 10 \
	--gpu ${device_id}

cp _tmp/smaps_sc/* ${testfolder}/
rm -rf _tmp

python predict_stage2.py \
	--cafferoot ${caffe_root}/ \
	--weights ./models/model_imagenet_bbox_pretrain_iter_780000_cascade_stage2_iter_80000.caffemodel \
	--model ./clarifai_stage2.prototxt \
	--blob fc8_msra10k \
	--testfolder ${testfolder}/ \
	--priorfolder ${testfolder}/ \
	--slic_n_segments 100 \
	--slic_compactness 10 \
	--gpu ${device_id}

cp _tmp/smaps_mc/* ${testfolder}/
rm -rf _tmp

