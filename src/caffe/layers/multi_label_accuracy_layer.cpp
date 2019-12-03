#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  prob_threshold_ = this->layer_param_.multi_label_accuracy_param().prob_threshold();
  has_ignore_label_ =
    this->layer_param_.multi_label_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.multi_label_accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.multi_label_accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions; ";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / (outer_num_ * inner_num_);
  const int num_labels = 2;
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
	  int binarycnt = 0;
	  int instancecnt = 0;
	  for (int j = 0; j< dim; ++j){
		  for (int k=0; k<inner_num_; ++k){
	 	     const int label_value =
        		  static_cast<int>(bottom_label[i*dim*inner_num_ + j * inner_num_ + k]);
		      if (has_ignore_label_ && label_value == ignore_label_) {
		        continue;
		      }
		      DCHECK_GE(label_value, 0);
		      DCHECK_LT(label_value, num_labels);
		      // check if it is a true label
		      if (top.size() > 1 && label_value == 1) ++nums_buffer_.mutable_cpu_data()[j];
		      if (bottom_data[i*dim*inner_num_ + j*inner_num_ + k] > prob_threshold_ && label_value == 1){
	         	 ++binarycnt;
	        	  if (top.size() > 1) ++top[1]->mutable_cpu_data()[j];
		      }
		      if (bottom_data[i*dim*inner_num_ + j*inner_num_ + k]< prob_threshold_ && label_value == 0 ) {
	         	 ++binarycnt;
	      		}
		   ++instancecnt;
		  }
	  }
	if(binarycnt == instancecnt){++accuracy;}
	++count;
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy /count ;
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MultiLabelAccuracy);

}  // namespace caffe
