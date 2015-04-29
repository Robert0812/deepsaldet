//
// Based on data_layer.cpp by Yangqing Jia.

#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <ctime>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

// caffe.proto > LayerParameter > WindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

// Thread fetching the data
template <typename Dtype>
void MCWindowDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = prefetch_data_.mutable_cpu_data();
  Dtype* top_mask = prefetch_mask_.mutable_cpu_data();
  Dtype* top_label1 = prefetch_label1_.mutable_cpu_data();
  //Dtype* top_label2 = prefetch_label2_.mutable_cpu_data();
  const Dtype scale = this->layer_param_.mcwindow_data_param().scale();
  const int batch_size = this->layer_param_.mcwindow_data_param().batch_size();
  const int crop_size = this->layer_param_.mcwindow_data_param().crop_size();
  const int mask_crop_size = this->layer_param_.mcwindow_data_param().mask_crop_size();
  const int context_pad = this->layer_param_.mcwindow_data_param().context_pad();
  const bool mirror = this->layer_param_.mcwindow_data_param().mirror();
  const float fg_fraction =
      this->layer_param_.mcwindow_data_param().fg_fraction();
  const Dtype* mean = data_mean_.cpu_data();
  const int mean_off = (data_mean_.width() - crop_size) / 2;
  const int mean_width = data_mean_.width();
  const int mean_height = data_mean_.height();
  cv::Size cv_crop_size(crop_size, crop_size);
  cv::Size cv_mask_crop_size(mask_crop_size, mask_crop_size);
  const string& crop_mode = this->layer_param_.mcwindow_data_param().crop_mode();
  bool shuffle = this->layer_param_.mcwindow_data_param().shuffle();
  // bool is_mask = this->layer_param_.mcwindow_data_param().is_mask();
  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(prefetch_data_.count(), Dtype(0), top_data);
  caffe_set(prefetch_mask_.count(), Dtype(0), top_mask);

  // LOG(INFO) << "what is count: " << prefetch_data_.count();

  const int num_fg = static_cast<int>(static_cast<float>(batch_size)
      * fg_fraction);
  const int num_samples[2] = { batch_size - num_fg, num_fg };

  int item_id = 0;
  int total_window = all_windows_.size();
  // sample from bg set then fg set
  for (int is_fg = 0; is_fg < 2; ++is_fg) {
    for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
      vector<float> window;
      if (shuffle) {
      	// sample a window
      	const unsigned int rand_index = PrefetchRand();
      	window = (is_fg) ?
          fg_windows_[rand_index % fg_windows_.size()] :
          bg_windows_[rand_index % bg_windows_.size()];
      }
      else{
        cnt_ %= total_window;
        window = all_windows_[cnt_++];
      }

      bool do_mirror = false;
      if (mirror && PrefetchRand() % 2) {
        do_mirror = true;
      }

      // load the image containing the window
      pair<std::string, vector<int> > image =
          image_database_[window[MCWindowDataLayer<Dtype>::IMAGE_INDEX]];
      
      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);

      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
      }
      const int channels = cv_img.channels();

      // load the mask containing the window
      const int msk_channels = mask_database_[0].size();
 
      vector<cv::Mat> cv_msk_all;
      for (int i=0; i < msk_channels; i++){	
      	pair<std::string, vector<int> > mask = mask_database_[window[MCWindowDataLayer<Dtype>::IMAGE_INDEX]][i];
      	cv::Mat cv_msk_cur = cv::imread(mask.first, CV_LOAD_IMAGE_GRAYSCALE);
	       // LOG(INFO) << "id: " << MCWindowDataLayer<Dtype>::IMAGE_INDEX << "msk size: " << cv_msk_cur.size(); 
      	if (!cv_msk_cur.data) {
		      LOG(ERROR) << "Could not open or find file " << mask.first;
        	return;
      	}
  	     cv_msk_all.push_back(cv_msk_cur);
      }

      // crop window out of image and warp it
      int x1 = window[MCWindowDataLayer<Dtype>::X1];
      int y1 = window[MCWindowDataLayer<Dtype>::Y1];
      int x2 = window[MCWindowDataLayer<Dtype>::X2];
      int y2 = window[MCWindowDataLayer<Dtype>::Y2];

      int pad_w = 0;
      int pad_h = 0;
      if (context_pad > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size) /
            static_cast<Dtype>(crop_size - 2*context_pad);

        // compute the expanded region
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);

        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;
        // if we're mirroring, we mirror the padding too (to be pedantic)
        if (do_mirror) {
          pad_w = pad_x2;
        } else {
          pad_w = pad_x1;
        }

        // ensure that the warped, clipped region plus the padding fits in the
        // crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size) {
          cv_crop_size.height = crop_size - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size) {
          cv_crop_size.width = crop_size - pad_w;
        }
      }
      
      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);
      
      cv::Rect msk_roi(x1, y1, x2-x1+1, y2-y1+1);
      vector<cv::Mat> cv_cropped_msk_all;
      for (int i=0; i<msk_channels; i++){
      	cv::Mat cv_cropped_msk = cv_msk_all[i](msk_roi);
      	cv::resize(cv_cropped_msk, cv_cropped_msk,
      	  cv_mask_crop_size, 0, 0, cv::INTER_LINEAR);
	         cv_cropped_msk_all.push_back(cv_cropped_msk);
      }

      // get window label
      top_label1[item_id] = window[MCWindowDataLayer<Dtype>::LABEL];
      
      //int obj_x1 = window[MCWindowDataLayer<Dtype>::OBJ_X1];
      //int obj_y1 = window[MCWindowDataLayer<Dtype>::OBJ_Y1];
      //int obj_x2 = window[MCWindowDataLayer<Dtype>::OBJ_X2];
      //int obj_y2 = window[MCWindowDataLayer<Dtype>::OBJ_Y2];
      //int w0 = x2 - x1;
      //int h0 = y2 - y1;

      //float norm_x1 = 1.0*obj_x1/w0;
      //float norm_y1 = 1.0*obj_y1/h0;
      //float norm_x2 = 1.0*obj_x2/w0;
      //float norm_y2 = 1.0*obj_y2/h0;

      // horizontal flip at random
      if (do_mirror) {
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
	for (int i=0; i<msk_channels; i++) {
	  cv::flip(cv_cropped_msk_all[i], cv_cropped_msk_all[i], 1);
	}
	//norm_x1 = 1.0 - norm_x2;
	//norm_x2 = 1.0 - norm_x1;
      }
      
      //top_label2[item_id*4] = norm_x1;
      //top_label2[item_id*4+1] = norm_y1;
      //top_label2[item_id*4+2] = norm_x2;
      //top_label2[item_id*4+3] = norm_y2;

      // copy the warped window into top_data
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel =
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);
            top_data[((item_id * channels + c) * crop_size + h + pad_h)
                     * crop_size + w + pad_w]
                = (pixel
                    - mean[(c * mean_height + h + mean_off + pad_h)
                           * mean_width + w + mean_off + pad_w])
                  * scale;
	    //Dtype temp =  (pixel - mean[(c * mean_height + h + mean_off + pad_h)
            //               * mean_width + w + mean_off + pad_w])* scale;

	    //LOG(INFO) << "pixel value of image: " << temp;
          }
        }
      }

      // copy the warped mask window into top_mask
      for (int c=0; c<msk_channels; ++c){
      	for (int h = 0; h < cv_cropped_msk_all[c].rows; ++h) {
          for (int w = 0; w < cv_cropped_msk_all[c].cols; ++w) {
            Dtype pixel =
              	static_cast<Dtype>(cv_cropped_msk_all[c].at<uchar>(h, w));
            top_mask[((item_id * msk_channels + c) *  mask_crop_size + h + pad_h)
                   	* mask_crop_size + w + pad_w]
              = pixel/255.0;
	  }
        }
      }
	
      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[WindowDataLayer<Dtype>::X1]+1 << std::endl
          << window[WindowDataLayer<Dtype>::Y1]+1 << std::endl
          << window[WindowDataLayer<Dtype>::X2]+1 << std::endl
          << window[WindowDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[item_id] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((item_id * channels + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      item_id++;
    }
  }
}

template <typename Dtype>
MCWindowDataLayer<Dtype>::~MCWindowDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void MCWindowDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  // SetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Multichannel Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.mcwindow_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.mcwindow_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.mcwindow_data_param().fg_fraction();

  std::ifstream infile(this->layer_param_.mcwindow_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.mcwindow_data_param().source() << std::endl;
 
  // bool is_mask = this->layer_param_.mcwindow_data_param().is_mask();

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels, msk_channels;
  if (!(infile >> hashtag >> image_index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path, mask_path_cur; 
    vector<string> mask_path;
    infile >> image_path;
    infile >> msk_channels;
    for (int i=0; i < msk_channels; i++){	
    	infile >> mask_path_cur;
	mask_path.push_back(mask_path_cur);
    }
    // read image dimensions
    vector<int> image_size(3);
    vector<int> mask_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    
    mask_size[0] = 1;
    mask_size[1] = image_size[1];
    mask_size[2] = image_size[2];
    image_database_.push_back(std::make_pair(image_path, image_size));
    vector<std::pair<std::string, vector<int> > > mask_channels_;
    for (int i=0; i < msk_channels; i++){
    	//LOG(INFO) << "test here:" << mask_path[i] << "idx: " << i << "c:" << msk_channels; 
    	mask_channels_.push_back(std::make_pair(mask_path[i], mask_size));
    }
    mask_database_.push_back(mask_channels_);

    // read each box
    int num_windows;
    infile >> num_windows;
    const float fg_threshold =
        this->layer_param_.mcwindow_data_param().fg_threshold();
    const float bg_threshold =
        this->layer_param_.mcwindow_data_param().bg_threshold();
    for (int i = 0; i < num_windows; ++i) {
      int label, x1, y1, x2, y2;
      float overlap;
      infile >> label >> overlap >> x1 >> y1 >> x2 >> y2;
      vector<float> window(MCWindowDataLayer::NUM);
      window[MCWindowDataLayer::IMAGE_INDEX] = image_index;
      window[MCWindowDataLayer::LABEL] = label;
      window[MCWindowDataLayer::OVERLAP] = overlap;
      window[MCWindowDataLayer::X1] = x1;
      window[MCWindowDataLayer::Y1] = y1;
      window[MCWindowDataLayer::X2] = x2;
      window[MCWindowDataLayer::Y2] = y2;

      // add window to foreground list or background list and also add window to
      // a unified list
      if (overlap >= fg_threshold) {
        int label = window[MCWindowDataLayer::LABEL];
        CHECK_GT(label, 0);
        fg_windows_.push_back(window);
        label_hist.insert(std::make_pair(label, 0));
        label_hist[label]++;
      } else if (overlap < bg_threshold) {
        // background window, force label and overlap to 0
        window[MCWindowDataLayer::LABEL] = 0;
        window[MCWindowDataLayer::OVERLAP] = 0;
        bg_windows_.push_back(window);
        label_hist[0]++;
      }
      all_windows_.push_back(window);
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;
  LOG(INFO) << "Number of windows: " << all_windows_.size();

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  // reset the counter
  cnt_ = 0;

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.mcwindow_data_param().context_pad();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.mcwindow_data_param().crop_mode();

  // image
  int crop_size = this->layer_param_.mcwindow_data_param().crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.mcwindow_data_param().batch_size();
  (*top)[0]->Reshape(batch_size, channels, crop_size, crop_size);
  prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
  
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // mask
  int mask_crop_size = this->layer_param_.mcwindow_data_param().mask_crop_size();
  CHECK_GT(mask_crop_size, 0);
  (*top)[1]->Reshape(batch_size, msk_channels, mask_crop_size, mask_crop_size);
  prefetch_mask_.Reshape(batch_size, msk_channels, mask_crop_size, mask_crop_size);

  LOG(INFO) << "output mask size: " << (*top)[1]->num() << ","
      << (*top)[1]->channels() << "," << (*top)[1]->height() << ","
      << (*top)[1]->width();

  // label1
  (*top)[2]->Reshape(batch_size, 1, 1, 1);
  prefetch_label1_.Reshape(batch_size, 1, 1, 1);
  LOG(INFO) << "output label1 size: " << (*top)[2]->num() << ","
      << (*top)[2]->channels() << "," << (*top)[2]->height() << ","
      << (*top)[2]->width();

  // label2
  //(*top)[3]->Reshape(batch_size, 4, 1, 1);
  //prefetch_label2_.Reshape(batch_size, 4, 1, 1);
  //LOG(INFO) << "output label2 size: " << (*top)[3]->num() << ","
  //    << (*top)[3]->channels() << "," << (*top)[3]->height() << ","
  //    << (*top)[3]->width();  

  // check if we want to have mean
  if (this->layer_param_.mcwindow_data_param().has_mean_file()) {
    const string& mean_file =
        this->layer_param_.mcwindow_data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, channels, crop_size, crop_size);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_.mutable_cpu_data();
  prefetch_mask_.mutable_cpu_data();
  prefetch_label1_.mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void MCWindowDataLayer<Dtype>::CreatePrefetchThread() {
  
  // bool global_rand = this->layer_param_.mcwindow_data_param().global_rand();
  const bool prefetch_needs_rand =
      this->layer_param_.mcwindow_data_param().mirror() ||
      this->layer_param_.mcwindow_data_param().crop_size();
  if (prefetch_needs_rand) {

    // if (global_rand) {
    //   time_t now = time(0);
    //   tm *ltm = localtime (&now);
    //   const unsigned int prefetch_rng_seed = ltm->tm_mday*1000000 + ltm->tm_hour*10000 + ltm->tm_min*100 + ltm->tm_sec + ltm->tm_sec%2;
    //   //LOG(INFO) << "seed: " << prefetch_rng_seed;
    //   prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    // }
    // else {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    // }
    
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!StartInternalThread()) << "Pthread execution failed.";
}

template <typename Dtype>
void MCWindowDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!WaitForInternalThreadToExit()) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int MCWindowDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype MCWindowDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
             (*top)[0]->mutable_cpu_data());
  caffe_copy(prefetch_mask_.count(), prefetch_mask_.cpu_data(),
             (*top)[1]->mutable_cpu_data());
  caffe_copy(prefetch_label1_.count(), prefetch_label1_.cpu_data(),
             (*top)[2]->mutable_cpu_data());
  //caffe_copy(prefetch_label2_.count(), prefetch_label2_.cpu_data(),
  //           (*top)[3]->mutable_cpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(MCWindowDataLayer, Forward);
#endif

INSTANTIATE_CLASS(MCWindowDataLayer);

}  // namespace caffe

