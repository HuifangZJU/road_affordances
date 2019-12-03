#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

const int kNumClass = 3;
const int posDim = 14;
const int posScale = 16;

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  float imglabel;
  float woffset;
  float hoffset;
  float dis;
  float keyu;
  float keyv;
  while (infile >> filename >> imglabel >> woffset >> hoffset >> dis >> keyu >> keyv) {
    vector<float> label;
    label.push_back(imglabel);
    label.push_back(woffset);
    label.push_back(hoffset);
    label.push_back(dis);
    label.push_back(keyu);
    label.push_back(keyv);
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data().Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  int shape[4] = {batch_size,2*kNumClass+1,1,1};
  vector<int> label_shape(4,0);
  memcpy(&label_shape[0],shape,sizeof(shape));
  top[1]->Reshape(label_shape);
  LOG(INFO)<< "label_shape"<< label_shape.size();

  int pos[4] = {batch_size,kNumClass,posDim,posDim};
  vector<int> pos_shape(4,0);
  memcpy(&pos_shape[0],pos,sizeof(pos));
  top[2]->Reshape(pos_shape);
  LOG(INFO)<< "position_shape"<< pos_shape.size();
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label(0).Reshape(label_shape);
    this->prefetch_[i].label(1).Reshape(pos_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data().count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data().Reshape(top_shape);

  Dtype* prefetch_data = batch->data().mutable_cpu_data();
  Dtype* prefetch_label = batch->label(0).mutable_cpu_data();
  Dtype* prefetch_pos = batch->label(1).mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data().offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    //Huifang modified
    int label = int(lines_[lines_id_].second[0]);
    float distance = lines_[lines_id_].second[3];
    //if((label == 1 && lines_id_%4 != 1) || (label>1 && distance == 0)){
    if(label == 1 && lines_id_%2 != 1){
    	    lines_id_++;
	    item_id--;
    	if (lines_id_ >= lines_size) {
      	// We have reached the end. Restart from the first.
      	DLOG(INFO) << "Restarting data prefetching from start.";
      	lines_id_ = 0;
      	if (this->layer_param_.image_data_param().shuffle()) {
        	ShuffleImages();
      	}
   	 }	
	    continue;
    }
    int col_offset = int(lines_[lines_id_].second[1]);
    int row_offset = int(lines_[lines_id_].second[2]);
    int keyu = int(lines_[lines_id_].second[4]);
    int keyv = int(lines_[lines_id_].second[5]);
    //LOG(INFO) <<"keyuv before transform : "<<keyu<<"  "<<keyv;
    this->data_transformer_->Transform(cv_img,col_offset,row_offset,label,keyu,keyv,distance,&(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    int kNumPos = kNumClass*posDim*posDim;
    int numPos = posDim*posDim;
    keyu = int(keyu/posScale);    
    keyv = int(keyv/posScale);
    //binary label
    if(label<0){//negtive sample
    	prefetch_label[item_id*(2*kNumClass+1)] = kNumClass;
	label += 3;
    	for (int i=1;i<kNumClass+1;i++)
    	{
	       if(i == label + 1){
	    		prefetch_label[item_id*(kNumClass+2)+i]=0;
    			prefetch_label[item_id*(2*kNumClass+1)+kNumClass+i] = -1;
	    	}
	    	else{
	    		prefetch_label[item_id*(kNumClass+2)+i]=2;
    			prefetch_label[item_id*(2*kNumClass+1)+kNumClass+i] = -1;
	   	 }
    	}
    	prefetch_label[item_id*(kNumClass+2)+kNumClass+1] = -1;
    	for (int i = 0;i< kNumClass;i++)
   	 {
		 if(i == label){
		 	for(int j =0;j<numPos;j++){
		    		prefetch_pos[item_id*kNumPos + i*numPos + j] = 0;
		 	}
		 }
		 else{
		 	for(int j =0;j<numPos;j++){
		    		prefetch_pos[item_id*kNumPos + i*numPos + j] = 2;
		 	}
		 }
   	 }
	//label = kNumClass;
    }
    else if(label < kNumClass){//distance
    	prefetch_label[item_id*(2*kNumClass+1)] = label;
    	for (int i=1;i<kNumClass+1;i++)
    	{
		    if(i == label+1){
	    		prefetch_label[item_id*(2*kNumClass+1)+i]=1;
    			prefetch_label[item_id*(2*kNumClass+1)+kNumClass+i] = distance;
	    	}
	    	else{
	    		prefetch_label[item_id*(2*kNumClass+1)+i]=0;
    			prefetch_label[item_id*(2*kNumClass+1)+kNumClass+i] = -1;
	   	 }
    	}
    	for (int i = 0;i< kNumPos;i++)
   	 {
		    prefetch_pos[item_id*kNumPos + i] = 0;
   	 }
    	prefetch_pos[item_id*kNumPos + label* numPos + keyv* posDim + keyu] = 1;
    }
    else {//positive
	label -=3;
    	prefetch_label[item_id*(2*kNumClass+1)] = label;
    	for (int i=1;i<kNumClass+1;i++)
    	{
		    if(i == label+1){
	    		prefetch_label[item_id*(2*kNumClass+1)+i]=1;
    			prefetch_label[item_id*(2*kNumClass+1)+kNumClass+i] =  -1;
	    	}
	    	else{
	    		prefetch_label[item_id*(2*kNumClass+1)+i]=0;
    			prefetch_label[item_id*(2*kNumClass+1)+kNumClass+i] = -1;
	   	 }
    	}
    	for (int i = 0;i< kNumPos;i++)
   	 {
		    prefetch_pos[item_id*kNumPos + i] = 0;
   	 }
    	prefetch_pos[item_id*kNumPos + label* numPos + keyv* posDim + keyu] = 1;
    }
    
    //softmax label
    //LOG(INFO) <<"keyuv after transform : "<<keyu<<"  "<<keyv;

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
