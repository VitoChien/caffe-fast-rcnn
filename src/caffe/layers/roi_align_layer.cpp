// ------------------------------------------------------------------
// Project: Multi Person Parser
// Written by Tianrui Hui
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // TODO: roi_align_param在哪实现?
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  // 经过Pooling后的feature map的高
  pooled_height_ = roi_align_param.pooled_h();
  // 经过Pooling后的feature map的宽
  pooled_width_ = roi_align_param.pooled_w();
  // feature map与输入图片之间的比值(默认为1/16)，这个feature map指roi pooling层的输入
  spatial_scale_ = roi_align_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 输入的feature map的channel数
  channels_ = bottom[0]->channels();
  // 输入的feature map的高
  height_ = bottom[0]->height();
  // 输入的feature map的宽
  width_ = bottom[0]->width();
  // 设置输出的形状(N,C,H,W)，N=ROI的个数，C=channels_，H=pooled_height_，W=pooled_width_
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  // max_idx_h的形状与top一致，是pooling输出的每一个点对应回feature map中响应最大的点的h坐标
  // 数据类型应当是Dtype
  max_idx_h.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  // max_idx_w的形状与top一致，是pooling输出的每一个点对应回feature map中响应最大的点的h坐标
  // 数据类型应当是Dtype
  max_idx_w.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

// 模板类型Dtype，应该一般都是double类型
template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 输入有两部分组成，data和rois
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  Dtype* argmax_data_h = max_idx_h.mutable_cpu_data();
  caffe_set(top_count, Dtype(-1), argmax_data_h);
  Dtype* argmax_data_w = max_idx_w.mutable_cpu_data();
  caffe_set(top_count, Dtype(-1), argmax_data_w);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  // batch_index表示这个ROI属于哪张图片，一般就是0，因为1个batch只有1张图片
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    // 将ROI在原图上的坐标映射到feature map上
    // 注意这里不对坐标进行四舍五入了
    Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale_;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    //Util Values
    Dtype one = 1.0;
    Dtype zero = 0.0;

    // 计算当前这个roi在feature map上面的宽高
    Dtype roi_height = max(roi_end_h - roi_start_h, one);
    Dtype roi_width = max(roi_end_w - roi_start_w, one);

    // 计算pooling之后的feature map上的一个值对应于pooling之前的feature map上的大小
    // 注：由于roi的大小不一致，所以每次都需要计算一次
    // 要注意下面的Dtype实际上应该都是double，所以bin_size_h和bin_size_w都是浮点数
    // bin_size_h： ROI中划分的子窗口的高度
    const Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height_);
    // bin_size_h： ROI中划分的子窗口的宽度
    const Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width_);

    // 找到对应的roi的feature map，如果input data的batch size为1
    // 那么roi_batch_ind=0
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = h * roi_height / pooled_height_
          //  end (excluded) = (ph + 1) * roi_height / pooled_height_
          // 计算output上的一点对应于input上面区域的大小[hstart, wstart, hend, wend]
          // 把这个区域称作bin，论文中叫做sub window
          // 这里每个bin的顶点坐标都是浮点数了
          Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
          Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
          Dtype hend = static_cast<Dtype>(ph + 1)* bin_size_h;
          Dtype wend =static_cast<Dtype>(pw + 1) * bin_size_w;

          // 上面四个坐标是把ROI的左上角点当作(0,0)点来算的
          // 所以需要根据ROI左上角点在feature map中的实际坐标来平移一下
          // 同时这每个bin的h相关坐标要>=0且<=height_，w相关坐标要>=0且<=width_
          hstart = min(max(hstart + roi_start_h, zero), static_cast<Dtype>(height_));
          hend = min(max(hend + roi_start_h, zero), static_cast<Dtype>(height_));
          wstart = min(max(wstart + roi_start_w, zero), static_cast<Dtype>(width_));
          wend = min(max(wend + roi_start_w, zero), static_cast<Dtype>(width_));

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          // pool_index指的是此时计算的output的值对应于output的位置
          const int pool_index = ph * pooled_width_ + pw;
          // 如果bin矩形不符合，此处output的值设为0，此处的对应于输入区域的最大值坐标都为-1
          if (is_empty) {
            top_data[pool_index] = 0.0;
            argmax_data_h[pool_index] = -1.0;
            argmax_data_w[pool_index] = -1.0;
            continue;
          }

          // 遍历bin中点的步长
          Dtype h_stride = (hend - hstart)/ 3.0;
          Dtype w_stride = (wend - wstart)/ 3.0;

          Dtype maxval = -FLT_MAX;
          // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
          Dtype maxidx_h = -1;
          Dtype maxidx_w = -1;

          // 遍历output的值对应于input的区域块，才能取到input feature map上相应的响应值
          for (Dtype h = hstart+h_stride; h <= hend-h_stride+0.01; h += max(h_stride, Dtype(0.01))) {
            for (Dtype w = wstart+w_stride; w <= wend-w_stride+0.01; w += max(w_stride, Dtype(0.01))) {

              // (h_min, w_min): (h, w)所在的1x1方格中左上角点的整数坐标
              // (h_max, w_max): (h, w)所在的1x1方格中右下角点的整数坐标
              int h_min = static_cast<int>(floor(static_cast<Dtype>(h)));
              int w_min = static_cast<int>(floor(static_cast<Dtype>(w)));
              int h_max = static_cast<int>(ceil(static_cast<Dtype>(h)));
              int w_max = static_cast<int>(ceil(static_cast<Dtype>(w)));

              // 判断(h, w)所在的1x1方格中四个顶点是否都在feature map范围内
              bool is_up_left_in = w_min >= 0 && w_min <= width_ - 1
              && h_min >= 0 && h_min <= height_ - 1;
              bool is_up_right_in = w_max >= 0 && w_max <= width_ - 1
              && h_min >= 0 && h_min <= height_ - 1;
              bool is_down_left_in = w_min >= 0 && w_min <= width_ - 1
              && h_max >= 0 && h_max <= height_ - 1;
              bool is_down_right_in = w_max >= 0 && w_max <= width_ - 1
              && h_max >= 0 && h_max <= height_ - 1;

              // 算出(h, w)周围4个点在双线性插值中的权重
              Dtype w_left_up =(1 - (h - h_min)) * (1 - (w - w_min));
              Dtype w_right_down =(1 - (h_max - h)) * (1 - (w_max - w));
              Dtype w_left_down = (1 - (h_max - h)) * (1 - (w - w_min));
              Dtype w_right_up = (1 - (h - h_min)) * (1 - (w_max - w));

              Dtype val = 0.0;
              if(is_up_left_in)
                val += w_left_up * batch_data[h_min * width_ + w_min];
              if(is_up_right_in)
                val += w_right_up * batch_data[h_min * width_ + w_max];
              if(is_down_left_in)
                val += w_left_down * batch_data[h_max * width_ + w_min];
              if(is_down_right_in)
                val += w_right_down * batch_data[h_max * width_ + w_max];

              // val即为双线性插值得出的(h, w)点处feature map的响应值
              // 记录下最大的响应值val及其在bin(即feature map)中的坐标
              if (val > maxval) {
                maxval = val;
                maxidx_h = h;
                maxidx_w = w;
              }
            }
          }
          top_data[pool_index] = maxval;
          argmax_data_h[pool_index] = maxidx_h;
          argmax_data_w[pool_index] = maxidx_w;
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data_h += max_idx_h.offset(0, 1);
      argmax_data_w += max_idx_w.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
