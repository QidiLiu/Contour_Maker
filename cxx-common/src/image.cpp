// 图像 IO 与几何实现。预处理口径与 Python 侧 common/roi.py 严格一致，
// 这是 Python 权重与 MNN 推理结果能对齐的前提。
#include "cxxcommon/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>

#include "cxxcommon/png.hpp"

namespace cxxcommon {
namespace {

constexpr uint8_t kPadValue = 114;   // 与 Python letterbox 的填充值一致

inline uint8_t ClampU8(int v) {
  return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

}  // namespace

// ---------------------------------------------------------------- IO
// 图像 IO 走自带的 png.cpp（本项目只需 PNG，且目标平台无 libpng/zlib 开发头文件）。

bool LoadImage(const std::string& path, Image* out, std::string* err) {
  if (!out) return false;
  int w = 0, h = 0;
  std::vector<uint8_t> gray;
  if (!PngDecodeToGray(path, &w, &h, &gray, err)) return false;
  out->reset(w, h, 1);
  out->data.swap(gray);
  return true;
}

bool SaveImage(const std::string& path, const Image& img, std::string* err) {
  if (img.channels != 1) {
    if (err) *err = "SaveImage 目前仅支持单通道灰度";
    return false;
  }
  return PngEncodeGray(path, img.data.data(), img.width, img.height, err);
}

bool SaveMaskPng(const std::string& path, const std::vector<uint8_t>& mask,
                 int w, int h, std::string* err) {
  std::vector<uint8_t> img(mask.size());
  for (size_t i = 0; i < mask.size(); ++i) img[i] = mask[i] ? 255 : 0;
  return PngEncodeGray(path, img.data(), w, h, err);
}

// ---------------------------------------------------------------- 几何
Box ClipBox(const Box& b, int w, int h) {
  Box r;
  r.x1 = std::max(0.f, std::min(b.x1, static_cast<float>(w - 1)));
  r.y1 = std::max(0.f, std::min(b.y1, static_cast<float>(h - 1)));
  r.x2 = std::max(0.f, std::min(b.x2, static_cast<float>(w - 1)));
  r.y2 = std::max(0.f, std::min(b.y2, static_cast<float>(h - 1)));
  if (r.x2 < r.x1) std::swap(r.x1, r.x2);
  if (r.y2 < r.y1) std::swap(r.y1, r.y2);
  return r;
}

float BoxIou(const Box& a, const Box& b) {
  const float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
  const float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
  const float inter = iw * ih;
  const float ua = a.area() + b.area() - inter;
  return ua > 0.f ? inter / ua : 0.f;
}

// 双线性缩放（灰度）
static Image ResizeBilinear(const Image& src, int dst_w, int dst_h) {
  Image dst(dst_w, dst_h, 1);
  if (src.width <= 0 || src.height <= 0) return dst;
  const float sx = static_cast<float>(src.width) / dst_w;
  const float sy = static_cast<float>(src.height) / dst_h;
  for (int y = 0; y < dst_h; ++y) {
    const float fy = (y + 0.5f) * sy - 0.5f;
    const int y0 = std::max(0, std::min(src.height - 1, static_cast<int>(std::floor(fy))));
    const int y1 = std::max(0, std::min(src.height - 1, y0 + 1));
    const float wy = fy - std::floor(fy);
    for (int x = 0; x < dst_w; ++x) {
      const float fx = (x + 0.5f) * sx - 0.5f;
      const int x0 = std::max(0, std::min(src.width - 1, static_cast<int>(std::floor(fx))));
      const int x1 = std::max(0, std::min(src.width - 1, x0 + 1));
      const float wx = fx - std::floor(fx);
      const float v00 = src.at(y0, x0)[0], v10 = src.at(y0, x1)[0];
      const float v01 = src.at(y1, x0)[0], v11 = src.at(y1, x1)[0];
      const float top = v00 + (v10 - v00) * wx;
      const float bot = v01 + (v11 - v01) * wx;
      dst.at(y, x)[0] = ClampU8(static_cast<int>(std::lround(top + (bot - top) * wy)));
    }
  }
  return dst;
}

// 最近邻缩放（二值掩码）
static Image ResizeNearest(const Image& src, int dst_w, int dst_h) {
  Image dst(dst_w, dst_h, 1);
  if (src.width <= 0 || src.height <= 0) return dst;
  const float sx = static_cast<float>(src.width) / dst_w;
  const float sy = static_cast<float>(src.height) / dst_h;
  for (int y = 0; y < dst_h; ++y) {
    const int sy0 = std::max(0, std::min(src.height - 1, static_cast<int>(y * sy)));
    for (int x = 0; x < dst_w; ++x) {
      const int sx0 = std::max(0, std::min(src.width - 1, static_cast<int>(x * sx)));
      dst.at(y, x)[0] = src.at(sy0, sx0)[0];
    }
  }
  return dst;
}

Image Letterbox(const Image& src, int size, LetterboxInfo* info) {
  Image dst(size, size, 1);
  if (src.empty()) {
    if (info) { *info = LetterboxInfo{}; }
    return dst;
  }
  const float r = std::min(static_cast<float>(size) / src.height,
                           static_cast<float>(size) / src.width);
  int nw = std::max(1, static_cast<int>(std::lround(src.width * r)));
  int nh = std::max(1, static_cast<int>(std::lround(src.height * r)));
  const int pad_x = (size - nw) / 2;
  const int pad_y = (size - nh) / 2;

  Image resized = ResizeBilinear(src, nw, nh);
  for (int y = 0; y < nh; ++y) {
    for (int x = 0; x < nw; ++x) {
      dst.at(pad_y + y, pad_x + x)[0] = resized.at(y, x)[0];
    }
  }
  if (info) {
    info->ratio = r;
    info->pad_x = pad_x;
    info->pad_y = pad_y;
  }
  return dst;
}

bool CropRoi(const Image& src, const Box& box, Image* out) {
  if (!out) return false;
  const int x1 = static_cast<int>(std::floor(box.x1));
  const int y1 = static_cast<int>(std::floor(box.y1));
  const int x2 = static_cast<int>(std::ceil(box.x2));
  const int y2 = static_cast<int>(std::ceil(box.y2));
  if (x2 - x1 < 4 || y2 - y1 < 4) return false;
  out->reset(x2 - x1, y2 - y1, 1);
  for (int y = 0; y < out->height; ++y) {
    for (int x = 0; x < out->width; ++x) {
      (*out).at(y, x)[0] = src.at(y1 + y, x1 + x)[0];
    }
  }
  return true;
}

Image ResizeRoi(const Image& src, int size, bool square_pad) {
  if (src.empty()) return Image(size, size, 1);
  if (!square_pad) {
    return ResizeBilinear(src, size, size);
  }
  Image canvas(size, size, 1);
  for (auto& v : canvas.data) v = kPadValue;

  const float s = static_cast<float>(size) / std::max(src.width, src.height);
  const int nw = std::max(1, static_cast<int>(std::lround(src.width * s)));
  const int nh = std::max(1, static_cast<int>(std::lround(src.height * s)));
  Image r = ResizeBilinear(src, nw, nh);
  const int ox = (size - nw) / 2;
  const int oy = (size - nh) / 2;
  for (int y = 0; y < nh; ++y) {
    for (int x = 0; x < nw; ++x) {
      canvas.at(oy + y, ox + x)[0] = r.at(y, x)[0];
    }
  }
  return canvas;
}

void PasteRoiMask(const std::vector<uint8_t>& roi_mask, int roi_size,
                  const Box& roi_box, int orig_w, int orig_h,
                  std::vector<uint8_t>* out_mask) {
  out_mask->assign(static_cast<size_t>(orig_w) * orig_h, 0);
  if (roi_mask.empty()) return;

  const int rh = static_cast<int>(roi_box.height() + 0.5f);
  const int rw = static_cast<int>(roi_box.width() + 0.5f);
  if (rh <= 0 || rw <= 0) return;

  // 反解 square_pad 的居中偏移（与 ResizeRoi 一致）
  int y0 = 0, x0 = 0, nh = roi_size, nw = roi_size;
  {
    const float s = static_cast<float>(roi_size) / std::max(rh, rw);
    nw = std::max(1, static_cast<int>(std::lround(rw * s)));
    nh = std::max(1, static_cast<int>(std::lround(rh * s)));
    x0 = (roi_size - nw) / 2;
    y0 = (roi_size - nh) / 2;
  }

  const int x1 = static_cast<int>(std::floor(roi_box.x1));
  const int y1 = static_cast<int>(std::floor(roi_box.y1));

  for (int oy = 0; oy < rh; ++oy) {
    const int gy = y1 + oy;
    if (gy < 0 || gy >= orig_h) continue;
    for (int ox = 0; ox < rw; ++ox) {
      const int gx = x1 + ox;
      if (gx < 0 || gx >= orig_w) continue;
      // ROI 坐标 -> square_pad 坐标 -> 采样
      const int sxp = x0 + static_cast<int>(std::lround(ox * static_cast<double>(nw) / std::max(rw, 1)));
      const int syp = y0 + static_cast<int>(std::lround(oy * static_cast<double>(nh) / std::max(rh, 1)));
      if (sxp < 0 || sxp >= roi_size || syp < 0 || syp >= roi_size) continue;
      const uint8_t v = roi_mask[static_cast<size_t>(syp) * roi_size + sxp];
      if (v) (*out_mask)[static_cast<size_t>(gy) * orig_w + gx] = 1;
    }
  }
}

bool MaskToContour(const std::vector<uint8_t>& mask, int w, int h,
                   std::vector<std::vector<float>>* contour) {
  if (!contour || mask.empty()) return false;
  contour->clear();

  // 1) 取最大 8-连通域（对应 OpenCV RETR_EXTERNAL + max(contourArea) 的语义：
  //    只保留最大连通域的外边界，忽略内部孔洞)
  std::vector<uint8_t> visited(mask.size(), 0);
  std::vector<std::pair<int, int>> best;
  for (int start = 0; start < static_cast<int>(mask.size()); ++start) {
    if (!mask[start] || visited[start]) continue;
    std::vector<std::pair<int, int>> comp;
    std::vector<int> stack;
    stack.push_back(start);
    visited[start] = 1;
    while (!stack.empty()) {
      const int idx = stack.back();
      stack.pop_back();
      const int y = idx / w, x = idx % w;
      comp.emplace_back(x, y);
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy) continue;
          const int ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          const int nidx = ny * w + nx;
          if (visited[nidx] || !mask[nidx]) continue;
          visited[nidx] = 1;
          stack.push_back(nidx);
        }
      }
    }
    if (comp.size() > best.size()) best.swap(comp);
  }
  if (best.size() < 3) return false;

  // 2) 提取外边界像素（8-邻域中至少有一个非前景邻居的点），
  //    对应 OpenCV 的「边界」判定
  std::vector<std::pair<int, int>> border;
  border.reserve(best.size());
  for (const auto& p : best) {
    const int x = p.first, y = p.second;
    bool is_border = false;
    for (int dy = -1; dy <= 1 && !is_border; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy) continue;
        const int ny = y + dy, nx = x + dx;
        if (ny < 0 || ny >= h || nx < 0 || nx >= w) { is_border = true; break; }
        if (!mask[static_cast<size_t>(ny) * w + nx]) { is_border = true; break; }
      }
    }
    if (is_border) border.push_back(p);
  }
  if (border.size() < 3) {
    for (const auto& p : best) {
      contour->push_back({static_cast<float>(p.first), static_cast<float>(p.second)});
    }
    return true;
  }

  // 3) 按绕质心的极角排序成闭合多边形
  float cx = 0.f, cy = 0.f;
  for (const auto& p : border) { cx += p.first; cy += p.second; }
  cx /= static_cast<float>(border.size());
  cy /= static_cast<float>(border.size());

  std::sort(border.begin(), border.end(),
            [cx, cy](const std::pair<int, int>& a, const std::pair<int, int>& b) {
              const float aa = std::atan2(a.second - cy, a.first - cx);
              const float ba = std::atan2(b.second - cy, b.first - cx);
              if (aa != ba) return aa < ba;
              // 同角度时用距离打破平局，保证严格弱序
              const float da = (a.first - cx) * (a.first - cx) + (a.second - cy) * (a.second - cy);
              const float db = (b.first - cx) * (b.first - cx) + (b.second - cy) * (b.second - cy);
              if (da != db) return da > db;
              if (a.first != b.first) return a.first < b.first;
              return a.second < b.second;
            });

  contour->reserve(border.size());
  for (const auto& p : border) {
    contour->push_back({static_cast<float>(p.first), static_cast<float>(p.second)});
  }
  return contour->size() >= 3;
}

std::vector<uint8_t> GrayToNchwBlob(const Image& img) {
  // 返回原始字节（MNN 内部会按 needFloat 转换），这里仅做 NHWC->CHW 的重排
  const int c = img.channels;
  const size_t plane = static_cast<size_t>(img.width) * img.height;
  std::vector<uint8_t> out(plane * c);
  for (int ch = 0; ch < c; ++ch) {
    for (size_t i = 0; i < plane; ++i) {
      out[ch * plane + i] = img.data[i * c + ch];
    }
  }
  return out;
}

}  // namespace cxxcommon