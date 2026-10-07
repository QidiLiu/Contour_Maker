// 图像 IO 与基础几何。C++ 侧与 Python 的 common/roi.py 口径严格一致。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cxxcommon {

// ---------------------------------------------------------------- 图像
struct Image {
  int width = 0;
  int height = 0;
  int channels = 1;                 // 本项目统一单通道灰度
  std::vector<uint8_t> data;        // size = width*height*channels

  Image() = default;
  Image(int w, int h, int c = 1) { reset(w, h, c); }

  void reset(int w, int h, int c = 1) {
    width = w;
    height = h;
    channels = c;
    data.assign(static_cast<size_t>(w) * h * c, 0);
  }

  size_t bytes() const { return data.size(); }
  bool empty() const { return data.empty(); }

  uint8_t* at(int y, int x) { return &data[(static_cast<size_t>(y) * width + x) * channels]; }
  const uint8_t* at(int y, int x) const {
    return &data[(static_cast<size_t>(y) * width + x) * channels];
  }
  uint8_t at(int y, int x, int c) const { return data[(static_cast<size_t>(y) * width + x) * channels + c]; }
};

struct Box {
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  float width() const { return x2 - x1; }
  float height() const { return y2 - y1; }
  float area() const {
    float w = x2 - x1, h = y2 - y1;
    return w > 0 && h > 0 ? w * h : 0.f;
  }
};

// 一个带类别与轮廓的检测/分割结果
struct Instance {
  int class_id = -1;
  float score = 0.f;
  Box box;
  std::vector<std::vector<float>> contour;   // 原图坐标系, 每个点 (x,y)
  std::vector<uint8_t> mask;                 // 原图尺寸二值掩码 (0/1)
  int mask_w = 0, mask_h = 0;
  bool has_mask = false;
};

// ---------------------------------------------------------------- IO
// 用 stb_image 风格的最简 PNG/JPEG 解码（实现见 image_io.cpp）
bool LoadImage(const std::string& path, Image* out, std::string* err);
bool SaveImage(const std::string& path, const Image& img, std::string* err);
bool SaveMaskPng(const std::string& path, const std::vector<uint8_t>& mask,
                 int w, int h, std::string* err);

// ---------------------------------------------------------------- 几何
Box ClipBox(const Box& b, int w, int h);
float BoxIou(const Box& a, const Box& b);

struct LetterboxInfo {
  float ratio = 1.f;
  int pad_x = 0;
  int pad_y = 0;
};

// 等比缩放 + 居中填充（灰度）。pad 用 114，与 Python 侧一致。
Image Letterbox(const Image& src, int size, LetterboxInfo* info);

// ROI 裁剪（不含 pad_ratio 计算），失败返回 false
bool CropRoi(const Image& src, const Box& box, Image* out);

// 等比缩放到 size×size 后居中填充（square_pad=true），与 Python resize_roi 一致
Image ResizeRoi(const Image& src, int size, bool square_pad);

// 把 size×size 的二值掩码按 ROI 几何贴回原图
void PasteRoiMask(const std::vector<uint8_t>& roi_mask, int roi_size,
                  const Box& roi_box, int orig_w, int orig_h,
                  std::vector<uint8_t>* out_mask);

// 掩码 -> 最大外轮廓（原图坐标）。面积过小返回 false。
bool MaskToContour(const std::vector<uint8_t>& mask, int w, int h,
                   std::vector<std::vector<float>>* contour);

// 浮点 CHW -> NCHW float 张量数据（MNN 输入）
std::vector<uint8_t> GrayToNchwBlob(const Image& img);

}  // namespace cxxcommon