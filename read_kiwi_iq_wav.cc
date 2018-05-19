// -*- C++ -*-

#include <fstream>
#include <octave/oct.h>
#include <octave/oct-map.h>

class chunk_base {
public:
  std::string id() const { return std::string((char*)(_id), 4); }
  std::streampos size() const { return _size; }
private:
  int8_t   _id[4];
  uint32_t _size;
} __attribute__((packed));

class chunk_riff : public chunk_base {
public:
  std::string format() const { return std::string((char*)(_format), 4); }

private:
  int8_t _format[4];
} __attribute__((packed));

class chunk_fmt : public chunk_base {
public:
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
} __attribute__((packed));

class chunk_kiwi : public chunk_base {
public:
  uint8_t  last() const { return _last; }
  uint32_t gpssec() const { return _gpssec; }
  uint32_t gpsnsec() const { return _gpsnsec; }
private:
  uint8_t  _last, _dummy;
  uint32_t _gpssec, _gpsnsec;
} __attribute__((packed));

DEFUN_DLD (read_kiwi_iq_wav, args, nargout, "[d,sample_rate]=read_kiwi_wav(\"<wav file name\");")
{
  octave_value_list retval;

  const std::string filename = args(0).string_value();
  if (error_state)
    return retval;

  std::ifstream file(filename, std::ios::binary);

  octave_value_list cell_z, cell_last, cell_gpssec, cell_gpsnsec;

  chunk_base c;
  chunk_fmt fmt;

  int data_counter=0;
  while (file) {
    std::streampos pos = file.tellg();
    file.read((char*)(&c), sizeof(c));
    if (!file)
      break;

    if (c.id() == "RIFF") {
      chunk_riff cr;
      file.seekg(pos);
      file.read((char*)(&cr), sizeof(cr));
      if (cr.format() != "WAVE") {
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
      if (fmt.format() != 1 ||
          fmt.num_channels() != 2) {
        error("unsupported WAVE format");
        break;
      }
      retval(1) = fmt.sample_rate();
    } else if (c.id() == "data") {
      const int n = c.size()/4;
      ComplexNDArray a(dim_vector(n, 1));
      int16_t i=0, q=0;
      for (int j=0; j<n && file; ++j) {
        file.read((char*)(&i), sizeof(i));
        file.read((char*)(&q), sizeof(q));
        a(j) = std::complex<double>(i/32768., q/32768.);
      }
      cell_z(data_counter++) = a;
    } else if (c.id() == "kiwi") {
      file.seekg(pos);
      chunk_kiwi kiwi;
      file.read((char*)(&kiwi), sizeof(kiwi));
      cell_last(data_counter)    = kiwi.last();
      cell_gpssec(data_counter)  = kiwi.gpssec();
      cell_gpsnsec(data_counter) = kiwi.gpsnsec();
    } else {
      octave_stdout << "skipping unknown chunk " << c.id() << std::endl;
      pos = file.tellg();
      file.seekg(pos + c.size());
    }
  }
  octave_map map;
  map.setfield("z", cell_z);
  if (cell_last.length() == cell_z.length()) {
    map.setfield("gpslast", cell_last);
    map.setfield("gpssec",  cell_gpssec);
    map.setfield("gpsnsec", cell_gpsnsec);
  }
  retval(0) = map;

  return retval;
}
