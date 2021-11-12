#pragma once

#include "common.h"

namespace TreeSAT {
template <class _DataType> struct Tile {
    IMPORT_VALUETYPES(_DataType)

    Tile()
        : m_data(static_cast<DataType *>(nullptr)),
          m_tile_offset(UIntDataType::Zero()), m_prev_slice_xy(nullptr),
          m_prev_slice_yz(nullptr), m_prev_slice_xz(nullptr), m_tile_size(0) {}

    void tile_set_data(unsigned short tile_size, UIntDataType tile_offset,
                       DoubleDataType *prev_slice_xy,
                       DoubleDataType *prev_slice_yz,
                       DoubleDataType *prev_slice_xz, DataType tile_value) {
        m_tile_offset = tile_offset;
        m_data = tile_value;
        m_prev_slice_xy = prev_slice_xy;
        m_prev_slice_yz = prev_slice_yz;
        m_prev_slice_xz = prev_slice_xz;
        m_tile_size = tile_size;
    }

    void tile_set_data(unsigned short tile_size, UIntDataType tile_offset,
                       DoubleDataType *prev_slice_xy,
                       DoubleDataType *prev_slice_yz,
                       DoubleDataType *prev_slice_xz, DataType *tile_data) {
        m_tile_offset = tile_offset;
        m_data = tile_data;
        m_prev_slice_xy = prev_slice_xy;
        m_prev_slice_yz = prev_slice_yz;
        m_prev_slice_xz = prev_slice_xz;
        m_tile_size = tile_size;

        for (Index z = 1; z < m_tile_size; ++z) {
#pragma omp parallel for
            for (Index y = 0; y < m_tile_size; ++y) {
                for (Index x = 0; x < m_tile_size; ++x) {
                    set_tile_data_value(x, y, z,
                                        get_tile_data_value(x, y, z) +
                                            get_tile_data_value(x, y, z - 1));
                }
            }
        }
        for (Index y = 1; y < m_tile_size; ++y) {
#pragma omp parallel for
            for (Index z = 0; z < m_tile_size; ++z) {
                for (Index x = 0; x < m_tile_size; ++x) {
                    set_tile_data_value(x, y, z,
                                        get_tile_data_value(x, y, z) +
                                            get_tile_data_value(x, y - 1, z));
                }
            }
        }
        for (Index x = 1; x < m_tile_size; ++x) {
#pragma omp parallel for
            for (Index z = 0; z < m_tile_size; ++z) {
                for (Index y = 0; y < m_tile_size; ++y) {
                    set_tile_data_value(x, y, z,
                                        get_tile_data_value(x, y, z) +
                                            get_tile_data_value(x - 1, y, z));
                }
            }
        }
    }

    inline DataType get_tile_data_value(size_t x, size_t y, size_t z) const {
        if (m_data.index() == 0) { // 0 == DataType <-> sparse case
            return std::get<DataType>(m_data) * (x + 1) * (y + 1) * (z + 1);
        } else { // 1 == DataType* <-> dense case
            size_t ptr_offset =
                z * m_tile_size * m_tile_size + y * m_tile_size + x;
            return std::get<DataType *>(m_data)[ptr_offset];
        }
    }

    inline void set_tile_data_value(size_t x, size_t y, size_t z,
                                    DataType value) {
        size_t ptr_offset = z * m_tile_size * m_tile_size + y * m_tile_size + x;
        std::get<DataType *>(m_data)[ptr_offset] = value;
    }

    DoubleDataType get_value(size_t x, size_t y, size_t z) const {
        //            std::cout << "tile type " << m_type << " value " << x << "
        //            " << y << " " << z << " " << get_tile_data_value(x, y, z)
        //            << std::endl;
        return get_tile_data_value(x, y, z).template cast<double>() +
               m_prev_slice_xy[0]                       // - Sxy(x1, y1, z1)
               - m_prev_slice_yz[z * (m_tile_size + 1)] // + Syz( x1, y1, z2 )
               - m_prev_slice_xy[(y + 1) *
                                 (m_tile_size + 1)] // + Sxy( x1, y2, z1 )
               - m_prev_slice_xy[x + 1]             // + Sxy( x2, y1, z1 )
               + m_prev_slice_xy[(y + 1) * (m_tile_size + 1) + x +
                                 1]                   // - Sxy( x2, y2, z1 )
               + m_prev_slice_xz[z * m_tile_size + x] // - Sxz( x2, y1, z2 )
               + m_prev_slice_yz[z * (m_tile_size + 1) + y +
                                 1] // - Syz( x1, y2, z2 )
               + m_tile_offset.template cast<double>();
    }

    ~Tile() {
        delete[] m_prev_slice_xy;
        delete[] m_prev_slice_yz;
        delete[] m_prev_slice_xz;

        if (m_data.index() == 0) { // 0 == DataType <-> sparse case
            // do nothing
        } else { // 1 == DataType* <-> dense case
            auto data_ptr = std::get<DataType *>(m_data);
            delete[] data_ptr;
        }
    }

    //        std::string toString(unsigned int level) {
    //            std::ostringstream oss;
    //            for(unsigned int i = 0; i<level; ++i) oss << "    ";
    //            oss << value << " ["<<(int)splitplane << "]: "<<split;
    //            if(!isLeaf()){
    //                oss << std::endl;
    //                oss << children[0].toString(level+1) << std::endl;
    //                oss << children[1].toString(level+1) << std::endl;
    //            }
    //            return oss.str();
    //        }
    //
    size_t getSize() const {
        size_t size =
            sizeof(m_tile_offset) + 3 * sizeof(DoubleDataType *) +
            sizeof(DoubleDataType) *
                ((m_tile_size + 1) * (m_tile_size + 1) +
                 (m_tile_size + 1) * m_tile_size + m_tile_size * m_tile_size);

        if (m_data.index() == 0) { // 0 == DataType <-> sparse case
            return size + sizeof(DataType);
        } else { // 1 == DataType* <-> dense case
            auto ptr = std::get<DataType *>(m_data);
            return size + sizeof(ptr) +
                   sizeof(DataType) * m_tile_size * m_tile_size * m_tile_size;
        }
    }

    std::variant<DataType, DataType *> m_data;
    UIntDataType m_tile_offset;
    DoubleDataType *m_prev_slice_xy; // (m_tile_size + 1) * (m_tile_size + 1)
    DoubleDataType *m_prev_slice_yz; // (m_tile_size + 1) * m_tile_size
    DoubleDataType *m_prev_slice_xz; // m_tile_size * m_tile_size
    unsigned short m_tile_size;
};

} // namespace TreeSAT