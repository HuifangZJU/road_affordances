#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultilogisticLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  multilogistic_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.multilogistic_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(multilogistic_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, multilogistic_axis_);
  inner_num_ = bottom[0]->count(multilogistic_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void MultilogisticLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  //copy bottom_data to top_data (size, orig data,target data)
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  //negation 
  caffe_set(scale_.count(), Dtype(-1), scale_data);
  caffe_mul(top[0]->count(), top_data, scale_data, top_data);
  // exponentiation
  caffe_exp<Dtype>(top[0]->count(), top_data, top_data);
  //plus one 
  caffe_set(scale_.count(), Dtype(1), scale_data);
  caffe_add(top[0]->count(), top_data, scale_data, top_data);
  //reciprocal 
  caffe_div(top[0]->count(), scale_data,top_data, top_data);
}

template <typename Dtype>
void MultilogisticLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  //copy top_diff to bottom_diff
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  //scale:1-top_data 
  caffe_set(scale_.count(), Dtype(1), scale_data);
  caffe_sub(top[0]->count(), scale_data, top_data, scale_data);
  // elementwise multiplication: top_data*scale
  caffe_mul(top[0]->count(), scale_data, top_data, scale_data);
  // elementwise multiplication: top_diff*scale
  caffe_mul(top[0]->count(), bottom_diff, scale_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(MultilogisticLayer);
#endif

INSTANTIATE_CLASS(MultilogisticLayer);
REGISTER_LAYER_CLASS(Multilogistic);
}  // namespace caffe
