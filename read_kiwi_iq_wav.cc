// -*- C++ -*-

#include <fstream>
#include <octave/oct.h>
#include <octave/oct-map.h>

#if defined(__GNUC__) && not defined(__MINGW32__)
#  define _PACKED __attribute__((__packed__))
#else
#  define _PACKED
#  define _USE_PRAGMA_PACK
#endif

#ifdef _USE_PRAGMA_PACK
#  pragma pack(push, 1)
#endif

#if __cplusplus != 201103L
# include <cassert>
# define static_assert(A,B) assert(A)
#endif

class chunk_base {
public:
  chunk_base()
    : _id()
    , _size() {
    static_assert(sizeof(chunk_base) == 8, "chunk_base has wrong packed size");
  }
  std::string id() const { return std::string((char*)(_id), 4); }
  std::streampos size() const { return _size; }
private:
  int8_t   _id[4];
  uint32_t _size;
} _PACKED;

class chunk_riff : public chunk_base {
public:
  chunk_riff()
    : _format() {
    static_assert(sizeof(chunk_riff) == 8+4, "chunk_riff has wrong packed size");
  }
  std::string format() const { return std::string((char*)(_format), 4); }

private:
  int8_t _format[4];
} _PACKED;

class chunk_fmt : public chunk_base {
public:
  chunk_fmt()
    : _format()
    , _num_channels()
    , _sample_rate()
    , _byte_rate()
    , _block_align()
    , _dummy() {
    static_assert(sizeof(chunk_fmt) == 8+16, "chunk_fmt has wrong packed size");
  }
  uint16_t format()       const { return _format; }
  uint16_t num_channels() const { return _num_channels; }
  uint32_t sample_rate()  const { return _sample_rate; }
  uint32_t byte_rate()    const { return _byte_rate; }
  uint16_t block_align()  const { return _block_align; }

protected:
  uint16_t _format;
  uint16_t _num_channels;
  uint32_t _sample_rate;
  uint32_t _byte_rate;
  uint16_t _block_align;
  uint16_t _dummy;
} _PACKED;

class chunk_kiwi : public chunk_base {
public:
  chunk_kiwi()
    : _last()
    , _dummy()
    , _gpssec()
    , _gpsnsec() {
    static_assert(sizeof(chunk_kiwi) == 8+10, "chunk_kiwi has wrong packed size");
  }
  uint8_t  last() const { return _last; }
  uint32_t gpssec() const { return _gpssec; }
  uint32_t gpsnsec() const { return _gpsnsec; }
private:
  uint8_t  _last, _dummy;
  uint32_t _gpssec, _gpsnsec;
} _PACKED;

#ifdef _USE_PRAGMA_PACK
#  pragma pack(pop)
#  undef _USE_PRAGMA_PACK
#endif

#undef _PACKED

DEFUN_DLD (read_kiwi_iq_wav, args, nargout, "[d,sample_rate]=read_kiwi_wav(\"<wav file name\");")
{
  octave_value_list retval;

  const std::string filename = args(0).string_value();

  std::ifstream file(filename.c_str(), std::ios::binary);
  if (!file)
    error("file '%s' does not exist", filename.c_str());

  octave_value_list cell_z, cell_last, cell_gpssec, cell_gpsnsec;

  chunk_base c;
  chunk_fmt fmt;

  int data_counter=0;
  while (file) {
    const std::streampos pos = file.tellg();
    file.read((char*)(&c), sizeof(c));
    if (!file) {
      // end of file
      break;
    }
    if (c.id() == "RIFF") {
      chunk_riff cr;
      file.seekg(pos);
      file.read((char*)(&cr), sizeof(cr));
      if (!file || cr.format() != "WAVE") {
        error("'WAVE' chunk expected");
        break;
      }
      const int n = (int(cr.size())-sizeof(chunk_riff)-4)/2074;
      cell_z.resize(n);
      cell_last.resize(n);
      cell_gpssec.resize(n);
      cell_gpsnsec.resize(n);
    } else if (c.id() == "fmt ") {
      file.seekg(pos);
      file.read((char*)(&fmt), sizeof(fmt));
      if (!file ||
          fmt.format() != 1 ||
          fmt.num_channels() != 2) {
        error("unsupported WAVE format");
        break;
      }
      retval(1) = fmt.sample_rate();
    } else if (c.id() == "data") {
      const int n = c.size()/4;
      FloatComplexNDArray a(dim_vector(n, 1));
      int16_t i=0, j=0, q=0;
      for (; j<n && file; ++j) {
        file.read((char*)(&i), sizeof(i));
        file.read((char*)(&q), sizeof(q));
        a(j) = std::complex<float>(i/32768.0f, q/32768.0f);
      }
      if (j != n)
        error("incomplete 'data' chunk");
      cell_z(data_counter++) = a;
    } else if (c.id() == "kiwi") {
      file.seekg(pos);
      chunk_kiwi kiwi;
      file.read((char*)(&kiwi), sizeof(kiwi));
      if (!file)
        error("incomplete 'kiwi' chunk");
      cell_last(data_counter)    = kiwi.last();
      cell_gpssec(data_counter)  = kiwi.gpssec();
      cell_gpsnsec(data_counter) = kiwi.gpsnsec();
    } else {
      octave_stdout << "skipping unknown chunk '" << c.id() << "'" << std::endl;
      file.seekg(file.tellg() + c.size());
    }
  }
  octave_map map;
  map.setfield("z", cell_z);
  if (cell_last.length() == cell_z.length()) {
    map.setfield("gpslast", cell_last);
    map.setfield("gpssec",  cell_gpssec);
    map.setfield("gpsnsec", cell_gpsnsec);
  } else {
    error("number of GNSS timestamps does not match number of IQ samples");
  }
  retval(0) = map;

  return retval;
}
