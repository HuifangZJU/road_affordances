#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultilogisticWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter multilogistic_param(this->layer_param_);
  multilogistic_param.set_type("Multilogistic");
  multilogistic_layer_ = LayerRegistry<Dtype>::CreateLayer(multilogistic_param);
  multilogistic_bottom_vec_.clear();
  multilogistic_bottom_vec_.push_back(bottom[0]);
  multilogistic_top_vec_.clear();
  multilogistic_top_vec_.push_back(&prob_);
  multilogistic_layer_->SetUp(multilogistic_bottom_vec_, multilogistic_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  has_unknow_label_ =
    this->layer_param_.loss_param().has_unknow_label();
  if (has_unknow_label_) {
    unknow_label_ = this->layer_param_.loss_param().unknow_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void MultilogisticWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  multilogistic_layer_->Reshape(multilogistic_bottom_vec_, multilogistic_top_vec_);
  multilogistic_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.multilogistic_param().axis());
  outer_num_ = bottom[0]->count(0, multilogistic_axis_);
  inner_num_ = bottom[0]->count(multilogistic_axis_ + 1);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must equal number of predictions for multilogistic loss; ";
  if (top.size() >= 2) {
    // multilogistic output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MultilogisticWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the logistic prob values.
  multilogistic_layer_->Forward(multilogistic_bottom_vec_, multilogistic_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int channels = bottom[0]->shape(multilogistic_axis_);
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
     for(int j = 0; j<channels; j++){
        if (has_ignore_label_ && j == ignore_label_) {
              continue;
        }
	for (int k = 0; k < inner_num_; k++) {
 	   const int label_value = static_cast<int>(label[i * dim + j* inner_num_ + k]);
	   if(label_value == 1){
           	loss -= log(std::max(prob_data[i * dim + j * inner_num_ + k],
                	           Dtype(FLT_MIN)));
	   }
	   else if (label_value == 0){
           	loss -= log(1-std::max(prob_data[i * dim + j * inner_num_ + k],
                	           Dtype(FLT_MIN)));
	   }
	   else if (has_unknow_label_ && label_value == unknow_label_){
           	//loss -= log(1-std::max(prob_data[i * dim + j * inner_num_ + k],
                //	           Dtype(FLT_MIN)));
		   continue;
	   }
	   else{
    		LOG(FATAL)<< " Label input of Multilogistic layer cannot larger than 1 : "<<label_value<<"   :  "<< unknow_label_;
	   }
           ++count;
        }
     }
  }
  if (normalize_ && count>0) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void MultilogisticWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int channels = bottom[0]->shape(multilogistic_axis_);
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for ( int j = 0; j < channels; j++){
          if (has_ignore_label_ && j  == ignore_label_) {
               for (int k = 0; k < inner_num_; k++) {
                     bottom_diff[i * dim + j * inner_num_ + k] = 0;
               }

           } else {
         	for (int k = 0; k < inner_num_; k++) {
             	   const int label_value = static_cast<int>(label[i * dim + j * inner_num_ + k]);
		   if(label_value == 1){
                   	bottom_diff[i * dim + j * inner_num_ + k] -= 1;
		   }
		   if(has_unknow_label_ && label_value == unknow_label_){
                   	bottom_diff[i * dim + j * inner_num_ + k] = 0;
			continue;
		   }

                   ++count;
                 }
           }
       }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_ && count>0) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultilogisticWithLossLayer);
#endif

INSTANTIATE_CLASS(MultilogisticWithLossLayer);
REGISTER_LAYER_CLASS(MultilogisticWithLoss);

}  // namespace caffe
