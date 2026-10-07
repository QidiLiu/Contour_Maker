// 自包含 PNG 编解码接口（8-bit 灰度/彩色解码，8-bit 灰度编码）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cxxcommon {

// 解码任意常见 PNG 为 8-bit 灰度（RGB 按 Rec.601 亮度转灰度）
bool PngDecodeToGray(const std::string& path, int* w, int* h,
                     std::vector<uint8_t>* gray, std::string* err);

// 编码 8-bit 灰度 PNG
bool PngEncodeGray(const std::string& path, const uint8_t* data, int w, int h,
                   std::string* err);

}  // namespace cxxcommon