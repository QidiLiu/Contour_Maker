// 自包含的最小 PNG 编解码器。
//
// 目标平台不保证有 libpng / zlib 开发头文件，而本项目只需处理
// 8-bit 灰度 / RGB / RGBA / 调色板 PNG，因此这里自带实现，
// 使 C++ 测试子项目零外部依赖（除 MNN 本身）。
//
// 解码：支持 bit depth 8 的 color type 0/2/3/6，含全部 5 种行滤波。
// 编码：输出 8-bit 灰度 PNG，deflate 采用 stored（未压缩）块 + zlib 包装，
//       文件略大但完全合规，避免引入 zlib。
#include "cxxcommon/png.hpp"

#include <cstring>
#include <fstream>

namespace cxxcommon {
namespace {

inline uint8_t ClampU8(int v) {
  return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

int PaethPredict(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

// ---------------------------------------------------------------- CRC32
struct Crc32Table {
  uint32_t t[256];
  Crc32Table() {
    for (uint32_t n = 0; n < 256; ++n) {
      uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      t[n] = c;
    }
  }
};
const Crc32Table kCrc;

uint32_t Crc32(const uint8_t* data, size_t len, uint32_t crc = 0) {
  crc = ~crc;
  for (size_t i = 0; i < len; ++i) crc = kCrc.t[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

// ---------------------------------------------------------------- Adler32
uint32_t Adler32(const uint8_t* data, size_t len) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < len; ++i) {
    a = (a + data[i]) % 65521;
    b = (b + a) % 65521;
  }
  return (b << 16) | a;
}

// ---------------------------------------------------------------- inflate
// 原始 DEFLATE 解码（RFC 1951）
class BitReader {
 public:
  BitReader(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  bool GetBit(int* b) {
    if (bitcnt_ == 0) {
      if (pos_ >= n_) return false;
      cur_ = d_[pos_++];
      bitcnt_ = 8;
    }
    *b = cur_ & 1;
    cur_ >>= 1;
    --bitcnt_;
    return true;
  }
  bool GetBits(int count, uint32_t* out) {
    uint32_t v = 0;
    for (int i = 0; i < count; ++i) {
      int b;
      if (!GetBit(&b)) return false;
      v |= static_cast<uint32_t>(b) << i;
    }
    *out = v;
    return true;
  }
  void AlignByte() { bitcnt_ = 0; }
  size_t pos() const { return pos_; }
  void set_pos(size_t p) { pos_ = p; }
  bool ReadBytes(uint8_t* dst, size_t n) {
    if (pos_ + n > n_) return false;
    std::memcpy(dst, d_ + pos_, n);
    pos_ += n;
    return true;
  }

 private:
  const uint8_t* d_;
  size_t n_;
  size_t pos_ = 0;
  uint8_t cur_ = 0;
  int bitcnt_ = 0;
};

// 规范霍夫曼表
struct Huffman {
  std::vector<uint16_t> counts;   // 每位长的码数
  std::vector<uint16_t> symbols;  // 按码长排序的符号

  void Build(const uint8_t* lengths, int n) {
    counts.assign(16, 0);
    for (int i = 0; i < n; ++i) counts[lengths[i]]++;
    counts[0] = 0;
    symbols.assign(n, 0);
    std::vector<uint16_t> offs(16, 0);
    for (int len = 1; len < 16; ++len) offs[len] = offs[len - 1] + counts[len - 1];
    for (int i = 0; i < n; ++i) {
      if (lengths[i]) symbols[offs[lengths[i]]++] = static_cast<uint16_t>(i);
    }
  }

  bool Decode(BitReader* br, int* out) const {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; ++len) {
      int b;
      if (!br->GetBit(&b)) return false;
      code |= b;
      const int cnt = counts[len];
      if (code - first < cnt) {
        *out = symbols[index + (code - first)];
        return true;
      }
      index += cnt;
      first = (first + cnt) << 1;
      code <<= 1;
    }
    return false;
  }
};

const uint16_t kLenBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27,
                               31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                               2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const uint16_t kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
                                193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
                                6145, 8193, 12289, 16385, 24577};
const uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                                8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

bool Inflate(const uint8_t* src, size_t src_len, std::vector<uint8_t>* out) {
  BitReader br(src, src_len);
  Huffman fixed_lit, fixed_dist;
  {
    uint8_t l[288];
    for (int i = 0; i < 144; ++i) l[i] = 8;
    for (int i = 144; i < 256; ++i) l[i] = 9;
    for (int i = 256; i < 280; ++i) l[i] = 7;
    for (int i = 280; i < 288; ++i) l[i] = 8;
    fixed_lit.Build(l, 288);
    uint8_t d[30];
    for (int i = 0; i < 30; ++i) d[i] = 5;
    fixed_dist.Build(d, 30);
  }

  while (true) {
    int final_bit = 0;
    uint32_t type = 0;
    if (!br.GetBit(&final_bit)) return false;
    if (!br.GetBits(2, &type)) return false;

    // 注意: 必须先解码本块数据，再根据 final_bit 退出。
    // 若在解码前 break，最后一块的数据会被整块丢弃（解压结果静默截断）。
    if (type == 0) {                       // stored
      br.AlignByte();
      size_t p = br.pos();
      if (p + 4 > src_len) return false;
      const uint16_t len = static_cast<uint16_t>(src[p] | (src[p + 1] << 8));
      const uint16_t nlen = static_cast<uint16_t>(src[p + 2] | (src[p + 3] << 8));
      if (static_cast<uint16_t>(~len) != nlen) return false;
      p += 4;
      if (p + len > src_len) return false;
      out->insert(out->end(), src + p, src + p + len);
      br.set_pos(p + len);
    } else if (type == 1 || type == 2) {   // fixed / dynamic Huffman
      Huffman lit, dist;
      const Huffman* plit = &fixed_lit;
      const Huffman* pdist = &fixed_dist;
      if (type == 2) {
        int hlit, hdist, hclen;
        if (!br.GetBits(5, reinterpret_cast<uint32_t*>(&hlit)) ||
            !br.GetBits(5, reinterpret_cast<uint32_t*>(&hdist)) ||
            !br.GetBits(4, reinterpret_cast<uint32_t*>(&hclen))) {
          return false;
        }
        hlit += 257;
        hdist += 1;
        hclen += 4;
        static const int kOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4,
                                       12, 3, 13, 2, 14, 1, 15};
        uint8_t cl[19];
        std::memset(cl, 0, sizeof(cl));
        for (int i = 0; i < hclen; ++i) {
          int v;
          if (!br.GetBits(3, reinterpret_cast<uint32_t*>(&v))) return false;
          cl[kOrder[i]] = static_cast<uint8_t>(v);
        }
        Huffman clh;
        clh.Build(cl, 19);
        std::vector<uint8_t> lens(hlit + hdist, 0);
        int i = 0;
        while (i < hlit + hdist) {
          int sym;
          if (!clh.Decode(&br, &sym)) return false;
          if (sym < 16) {
            lens[i++] = static_cast<uint8_t>(sym);
          } else if (sym == 16) {
            if (i == 0) return false;
            uint32_t r;
            if (!br.GetBits(2, &r)) return false;
            const uint8_t prev = lens[i - 1];
            for (uint32_t k = 0; k < r + 3 && i < hlit + hdist; ++k) lens[i++] = prev;
          } else if (sym == 17) {
            uint32_t r;
            if (!br.GetBits(3, &r)) return false;
            for (uint32_t k = 0; k < r + 3 && i < hlit + hdist; ++k) lens[i++] = 0;
          } else {
            uint32_t r;
            if (!br.GetBits(7, &r)) return false;
            for (uint32_t k = 0; k < r + 11 && i < hlit + hdist; ++k) lens[i++] = 0;
          }
        }
        lit.Build(lens.data(), hlit);
        dist.Build(lens.data() + hlit, hdist);
        plit = &lit;
        pdist = &dist;
      }
      while (true) {
        int sym;
        if (!plit->Decode(&br, &sym)) return false;
        if (sym < 256) {
          out->push_back(static_cast<uint8_t>(sym));
        } else if (sym == 256) {
          break;
        } else {
          const int li = sym - 257;
          if (li >= 29) return false;
          uint32_t extra = 0;
          if (kLenExtra[li] && !br.GetBits(kLenExtra[li], &extra)) return false;
          const int length = kLenBase[li] + static_cast<int>(extra);
          int dsym;
          if (!pdist->Decode(&br, &dsym)) return false;
          if (dsym >= 30) return false;
          extra = 0;
          if (kDistExtra[dsym] && !br.GetBits(kDistExtra[dsym], &extra)) return false;
          const size_t distance = kDistBase[dsym] + extra;
          if (distance > out->size()) return false;
          // 注意: 必须先把字节取到局部变量再 push_back。
          // 直接 push_back((*out)[start+k]) 在 vector 扩容时会对已失效的内存取引用(UB)，
          // 表现为解压结果被静默截断。
          const size_t start = out->size() - distance;
          for (int k = 0; k < length; ++k) {
            const uint8_t b = (*out)[start + k];
            out->push_back(b);
          }
        }
      }
    } else {
      return false;   // type 3 保留
    }
    if (final_bit) break;   // 本块已解码完毕
  }
  return true;
}

// zlib 包装的 inflate（跳过 2 字节头 + 4 字节 adler）
bool ZlibInflate(const uint8_t* src, size_t len, std::vector<uint8_t>* out) {
  if (len < 6) return false;
  const uint8_t cmf = src[0];
  if ((cmf & 0x0F) != 8) return false;
  if (((src[0] << 8) | src[1]) % 31 != 0) return false;
  if (src[1] & 0x20) return false;   // 预设字典，不支持
  return Inflate(src + 2, len - 2 - 4, out);
}

// ---------------------------------------------------------------- deflate(stored)
void DeflateStored(const std::vector<uint8_t>& in, std::vector<uint8_t>* out) {
  out->push_back(0x78);   // CM=8, CINFO=7
  out->push_back(0x01);   // FCHECK, 无预设字典, 最高压缩
  size_t pos = 0;
  while (pos < in.size()) {
    const size_t n = std::min<size_t>(65535, in.size() - pos);
    const bool final_block = (pos + n >= in.size());
    out->push_back(final_block ? 1 : 0);
    out->push_back(static_cast<uint8_t>(n & 0xFF));
    out->push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
    const uint16_t nlen = static_cast<uint16_t>(~static_cast<uint16_t>(n));
    out->push_back(static_cast<uint8_t>(nlen & 0xFF));
    out->push_back(static_cast<uint8_t>((nlen >> 8) & 0xFF));
    out->insert(out->end(), in.begin() + pos, in.begin() + pos + n);
    pos += n;
  }
  const uint32_t ad = Adler32(in.data(), in.size());
  out->push_back(static_cast<uint8_t>((ad >> 24) & 0xFF));
  out->push_back(static_cast<uint8_t>((ad >> 16) & 0xFF));
  out->push_back(static_cast<uint8_t>((ad >> 8) & 0xFF));
  out->push_back(static_cast<uint8_t>(ad & 0xFF));
}

// ---------------------------------------------------------------- 反滤波
bool Unfilter(std::vector<uint8_t>* raw, int w, int h, int bpp, size_t stride) {
  const size_t need = (stride + 1) * static_cast<size_t>(h);
  if (raw->size() < need) return false;
  std::vector<uint8_t> out(static_cast<size_t>(stride) * h);
  for (int y = 0; y < h; ++y) {
    const uint8_t ft = (*raw)[static_cast<size_t>(y) * (stride + 1)];
    const uint8_t* src = raw->data() + static_cast<size_t>(y) * (stride + 1) + 1;
    uint8_t* dst = out.data() + static_cast<size_t>(y) * stride;
    const uint8_t* up = y > 0 ? out.data() + static_cast<size_t>(y - 1) * stride : nullptr;
    for (size_t i = 0; i < stride; ++i) {
      const int a = i >= static_cast<size_t>(bpp) ? dst[i - bpp] : 0;
      const int b = up ? up[i] : 0;
      const int c = (up && i >= static_cast<size_t>(bpp)) ? up[i - bpp] : 0;
      int v = src[i];
      switch (ft) {
        case 0: break;
        case 1: v += a; break;
        case 2: v += b; break;
        case 3: v += (a + b) / 2; break;
        case 4: v += PaethPredict(a, b, c); break;
        default: return false;
      }
      // PNG 的行滤波运算是 mod 256 回绕，必须截断而非饱和，
      // 否则会出现 0 -> 255 之类的错误。
      dst[i] = static_cast<uint8_t>(v & 0xFF);
    }
  }
  raw->swap(out);
  return true;
}

}  // namespace

// ---------------------------------------------------------------- 解码
bool PngDecodeToGray(const std::string& path, int* w, int* h,
                     std::vector<uint8_t>* gray, std::string* err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err) *err = "无法打开文件: " + path;
    return false;
  }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
  static const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  if (buf.size() < 8 || std::memcmp(buf.data(), kSig, 8) != 0) {
    if (err) *err = "非 PNG 文件(签名不匹配): " + path;
    return false;
  }

  int bit_depth = 0, color_type = 0, interlace = 0;
  std::vector<uint8_t> idat, palette;
  size_t pos = 8;
  while (pos + 8 <= buf.size()) {
    const uint32_t len = (buf[pos] << 24) | (buf[pos + 1] << 16) |
                         (buf[pos + 2] << 8) | buf[pos + 3];
    const char* type = reinterpret_cast<const char*>(buf.data() + pos + 4);
    const uint8_t* data = buf.data() + pos + 8;
    if (pos + 12 + len > buf.size()) break;

    if (std::strncmp(type, "IHDR", 4) == 0 && len >= 13) {
      *w = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
      *h = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7];
      bit_depth = data[8];
      color_type = data[9];
      interlace = data[12];
    } else if (std::strncmp(type, "PLTE", 4) == 0) {
      palette.assign(data, data + len);
    } else if (std::strncmp(type, "IDAT", 4) == 0) {
      idat.insert(idat.end(), data, data + len);
    } else if (std::strncmp(type, "IEND", 4) == 0) {
      break;
    }
    pos += 12 + len;
  }

  if (*w <= 0 || *h <= 0) {
    if (err) *err = "PNG 头无效";
    return false;
  }
  if (bit_depth != 8) {
    if (err) *err = "仅支持 8-bit 位深，实际 " + std::to_string(bit_depth);
    return false;
  }
  if (interlace != 0) {
    if (err) *err = "不支持隔行扫描";
    return false;
  }

  int samples_per_pixel = 1;
  switch (color_type) {
    case 0: samples_per_pixel = 1; break;   // 灰度
    case 2: samples_per_pixel = 3; break;   // RGB
    case 3: samples_per_pixel = 1; break;   // 调色板索引
    case 4: samples_per_pixel = 2; break;   // 灰度+alpha
    case 6: samples_per_pixel = 4; break;   // RGBA
    default:
      if (err) *err = "不支持的颜色类型 " + std::to_string(color_type);
      return false;
  }

  const size_t stride = static_cast<size_t>(*w) * samples_per_pixel;
  std::vector<uint8_t> raw;
  if (!ZlibInflate(idat.data(), idat.size(), &raw)) {
    if (err) *err = "inflate 失败";
    return false;
  }
  if (!Unfilter(&raw, *w, *h, samples_per_pixel, stride)) {
    if (err) *err = "反滤波失败";
    return false;
  }

  gray->assign(static_cast<size_t>(*w) * *h, 0);
  for (int y = 0; y < *h; ++y) {
    const uint8_t* row = raw.data() + static_cast<size_t>(y) * stride;
    for (int x = 0; x < *w; ++x) {
      uint8_t v;
      switch (color_type) {
        case 0: v = row[x]; break;
        case 4: v = row[x * 2]; break;
        case 2: {
          const int r = row[x * 3], g = row[x * 3 + 1], b = row[x * 3 + 2];
          v = static_cast<uint8_t>((r * 299 + g * 587 + b * 114) / 1000);
          break;
        }
        case 6: {
          const int r = row[x * 4], g = row[x * 4 + 1], b = row[x * 4 + 2];
          v = static_cast<uint8_t>((r * 299 + g * 587 + b * 114) / 1000);
          break;
        }
        case 3: {
          const size_t idx = static_cast<size_t>(row[x]) * 3;
          if (idx + 2 < palette.size()) {
            const int r = palette[idx], g = palette[idx + 1], b = palette[idx + 2];
            v = static_cast<uint8_t>((r * 299 + g * 587 + b * 114) / 1000);
          } else {
            v = 0;
          }
          break;
        }
        default: v = 0; break;
      }
      (*gray)[static_cast<size_t>(y) * *w + x] = v;
    }
  }
  return true;
}

// ---------------------------------------------------------------- 编码
bool PngEncodeGray(const std::string& path, const uint8_t* data, int w, int h,
                   std::string* err) {
  if (w <= 0 || h <= 0) {
    if (err) *err = "尺寸无效";
    return false;
  }
  // 原始扫描线: 每行前置 filter type 0
  std::vector<uint8_t> raw;
  raw.reserve((static_cast<size_t>(w) + 1) * h);
  for (int y = 0; y < h; ++y) {
    raw.push_back(0);
    raw.insert(raw.end(), data + static_cast<size_t>(y) * w,
               data + static_cast<size_t>(y + 1) * w);
  }
  std::vector<uint8_t> z;
  DeflateStored(raw, &z);

  std::ofstream f(path, std::ios::binary);
  if (!f) {
    if (err) *err = "无法写入: " + path;
    return false;
  }

  static const uint8_t kSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  f.write(reinterpret_cast<const char*>(kSig), 8);

  auto put_u32 = [&f](uint32_t v) {
    uint8_t b[4] = {static_cast<uint8_t>(v >> 24), static_cast<uint8_t>(v >> 16),
                    static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v)};
    f.write(reinterpret_cast<const char*>(b), 4);
  };
  auto write_chunk = [&](const char* type, const uint8_t* payload, size_t len) {
    put_u32(static_cast<uint32_t>(len));
    std::vector<uint8_t> body;
    body.insert(body.end(), type, type + 4);
    body.insert(body.end(), payload, payload + len);
    f.write(reinterpret_cast<const char*>(body.data()),
            static_cast<std::streamsize>(body.size()));
    put_u32(Crc32(body.data(), body.size()));
  };

  std::vector<uint8_t> ihdr(13);
  ihdr[0] = static_cast<uint8_t>(w >> 24); ihdr[1] = static_cast<uint8_t>(w >> 16);
  ihdr[2] = static_cast<uint8_t>(w >> 8);  ihdr[3] = static_cast<uint8_t>(w);
  ihdr[4] = static_cast<uint8_t>(h >> 24); ihdr[5] = static_cast<uint8_t>(h >> 16);
  ihdr[6] = static_cast<uint8_t>(h >> 8);  ihdr[7] = static_cast<uint8_t>(h);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 0;   // color type: grayscale
  ihdr[10] = 0;  // compression
  ihdr[11] = 0;  // filter
  ihdr[12] = 0;  // interlace
  write_chunk("IHDR", ihdr.data(), ihdr.size());
  write_chunk("IDAT", z.data(), z.size());
  write_chunk("IEND", nullptr, 0);

  if (!f.good()) {
    if (err) *err = "写入失败: " + path;
    return false;
  }
  return true;
}

}  // namespace cxxcommon