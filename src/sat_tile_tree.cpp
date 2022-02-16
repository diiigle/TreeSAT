#include <algorithm>
#include <iostream>
#include <variant>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#ifdef SAT_TILE_TREE_BUILD_PYTHON
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

#include "bounding_box.h"
#include "common.h"
#include "tile.h"

// #define SAT_TILE_TREE_STATS

namespace TreeSAT {
template <typename _DataType> class SATTileTree {
    IMPORT_VALUETYPES(_DataType)
    using TensorType =
        Eigen::TensorMap<Eigen::Tensor<DataType, 3, Eigen::RowMajor>>;
    using TileTensor = Eigen::Tensor<Tile<DataType>, 3, Eigen::RowMajor>;

  public:
#ifdef SAT_TILE_TREE_BUILD_PYTHON
    using ArrayType = py::array_t<typename DataType::Scalar,
                                  py::array::c_style | py::array::forcecast>;

    SATTileTree(const ArrayType &volume, unsigned short tile_size)
        : m_tile_size(tile_size) {
        py::buffer_info buffer = volume.request();

        if (buffer.ndim != 4)
            throw std::runtime_error("Number of dimensions must be 4");

        if (buffer.shape[3] != Dimensionality) {
            std::ostringstream err;
            err << "Input array of shape [..., " << buffer.shape[3]
                << "] does not match Dimensionality " << Dimensionality;
            throw std::runtime_error(err.str());
        }
        initialize(reinterpret_cast<DataType *>(buffer.ptr), buffer.shape[2],
                   buffer.shape[1], buffer.shape[0]);
    }
#endif

    SATTileTree(DataType *data, Eigen::Index dimX, Eigen::Index dimY,
                Eigen::Index dimZ, unsigned short tile_size)
        : m_tile_size(tile_size) {
        initialize(data, dimX, dimY, dimZ);
    }

    ~SATTileTree() {}

    DataType queryAverage(const BoundingBox &queryBox) {
        DataType sum;
        if (m_dataBBox.contains(queryBox)) {
            sum = get_sum_from_sat(queryBox);
        } else {
            // handle virtual zeros around
            auto overlap_box = m_dataBBox.overlap(queryBox);

            if (overlap_box.volume() > 0) {
                sum = get_sum_from_sat(overlap_box);
            } else {
                return DataType::Zero();
            }
        }
        return sum / queryBox.volume();
    }

#ifdef SAT_TILE_TREE_BUILD_PYTHON
    DataType queryAverageSlice(py::slice x_slice, py::slice y_slice,
                               py::slice z_slice) {
        Index x1, x2, y1, y2, z1, z2, step, slicelength;
        x_slice.compute(m_dataBBox.upper[0], &x1, &x2, &step, &slicelength);
        y_slice.compute(m_dataBBox.upper[1], &y1, &y2, &step, &slicelength);
        z_slice.compute(m_dataBBox.upper[2], &z1, &z2, &step, &slicelength);

        BoundingBox queryBox({x1, y1, z1}, {x2, y2, z2});

        return queryAverage(queryBox);
    }

    ArrayType querySingularPy(Index x, Index y, Index z) {
        auto result = ArrayType({BoundingBox::Scalar(Dimensionality)});
        auto r = result.mutable_unchecked();
        auto value = get_sat_value(x, y, z);
        for (unsigned char c = 0; c < Dimensionality; ++c) {
            r(c) = value[c];
        }
        return result;
    }

    ArrayType convertDense() {
        vec3 dimensions = m_dataBBox.upper - m_dataBBox.lower;
        auto result = ArrayType({dimensions[2], dimensions[1], dimensions[0],
                                 BoundingBox::Scalar(Dimensionality)});
        auto r = result.mutable_unchecked();

        for (Index z = 0; z < dimensions[2]; ++z) {
#pragma omp parallel for
            for (Index y = 0; y < dimensions[1]; ++y) {
                for (Index x = 0; x < dimensions[0]; ++x) {
                    auto &tile = m_tiles_tensor(
                        z / m_tile_size, y / m_tile_size, x / m_tile_size);
                    auto value =
                        tile.get_value(x - (x / m_tile_size) * m_tile_size,
                                       y - (y / m_tile_size) * m_tile_size,
                                       z - (z / m_tile_size) * m_tile_size);
                    for (unsigned char c = 0; c < Dimensionality; ++c) {
                        r(z, y, x, c) = value[c];
                    }
                }
            }
        }
        return result;
    }
#endif

    size_t size() const {
        size_t size = sizeof(SATTileTree);

        vec3 volume_dimensions = m_dataBBox.extend();
        vec3 tile_tree_dimensions =
            (volume_dimensions +
             vec3(m_tile_size - 1, m_tile_size - 1, m_tile_size - 1)) /
            m_tile_size;
        for (Index tile_z = 0; tile_z < tile_tree_dimensions[2]; ++tile_z) {
            for (Index tile_y = 0; tile_y < tile_tree_dimensions[1]; ++tile_y) {
                for (Index tile_x = 0; tile_x < tile_tree_dimensions[0];
                     ++tile_x) {
                    auto &tile_tensor = m_tiles_tensor(tile_z, tile_y, tile_x);
                    size += tile_tensor.getSize();
                }
            }
        }
        return size;
    }

    Eigen::Vector4i shape() const {
        return Eigen::Vector4i(m_dataBBox.upper[2], m_dataBBox.upper[1],
                               m_dataBBox.upper[0], Dimensionality);
    };

  private:
    TileTensor m_tiles_tensor;
    BoundingBox m_dataBBox;
    unsigned short m_tile_size;
#ifdef SAT_TILE_TREE_STATS
    long int __STATS_requests = 0;
    long int __STATS_cache_hits = 0;
    long int __STATS_cache_collisions = 0;
#endif // SAT_TILE_TREE_STATS

    void initialize(DataType *data, Eigen::Index dimX, Eigen::Index dimY,
                    Eigen::Index dimZ) {
        auto volume_tensor = TensorType(data, dimZ, dimY, dimX);

        m_dataBBox = BoundingBox({0, 0, 0}, vec3(dimX, dimY, dimZ));

#ifdef SAT_TILE_TREE_STATS
        __STATS_requests = 0;
        __STATS_cache_hits = 0;
        __STATS_cache_collisions = 0;
#endif // SAT_TILE_TREE_STATS

        constructTree(m_dataBBox, volume_tensor);
    }

    void constructTree(const BoundingBox &dimensions, const TensorType &data) {
        vec3 volume_dimensions = dimensions.upper - dimensions.lower;
        vec3 tile_tree_dimensions =
            (volume_dimensions +
             vec3(m_tile_size - 1, m_tile_size - 1, m_tile_size - 1)) /
            m_tile_size;
        m_tiles_tensor =
            TileTensor(tile_tree_dimensions[2], tile_tree_dimensions[1],
                       tile_tree_dimensions[0]);

        DoubleDataType sat_tile_offset_double;
        UIntDataType sat_tile_offset;
        bool all_values_equal_flag;
        DataType first_tile_value;
        Index volume_index_offset_x, volume_index_offset_y,
            volume_index_offset_z;

        for (Index tile_z = 0; tile_z < tile_tree_dimensions[2]; ++tile_z) {
            for (Index tile_y = 0; tile_y < tile_tree_dimensions[1]; ++tile_y) {
                for (Index tile_x = 0; tile_x < tile_tree_dimensions[0];
                     ++tile_x) {
                    all_values_equal_flag = true;

                    volume_index_offset_x = tile_x * m_tile_size;
                    volume_index_offset_y = tile_y * m_tile_size;
                    volume_index_offset_z = tile_z * m_tile_size;

                    first_tile_value =
                        data(volume_index_offset_z, volume_index_offset_y,
                             volume_index_offset_x);

                    auto tile_data =
                        new DataType[m_tile_size * m_tile_size * m_tile_size];

                    for (Index z = 0; z < m_tile_size; ++z) {
                        for (Index y = 0; y < m_tile_size; ++y) {
                            for (Index x = 0; x < m_tile_size; ++x) {
                                DataType value;
                                if (z + volume_index_offset_z >=
                                        volume_dimensions[2] ||
                                    y + volume_index_offset_y >=
                                        volume_dimensions[1] ||
                                    x + volume_index_offset_x >=
                                        volume_dimensions[0]) {
                                    value = DataType::Zero();
                                } else {
                                    value = data(volume_index_offset_z + z,
                                                 volume_index_offset_y + y,
                                                 volume_index_offset_x + x);
                                    if (value != first_tile_value) {
                                        all_values_equal_flag = false;
                                    }
                                }
                                Index ptr_offset =
                                    z * m_tile_size * m_tile_size +
                                    y * m_tile_size + x;
                                tile_data[ptr_offset] = value;
                            }
                        }
                    }

                    if (tile_x > 0 && tile_y > 0 && tile_z > 0) {
                        sat_tile_offset_double =
                            m_tiles_tensor(tile_z - 1, tile_y - 1, tile_x - 1)
                                .get_value(m_tile_size - 1, m_tile_size - 1,
                                           m_tile_size - 1);
                    } else {
                        sat_tile_offset_double = DoubleDataType::Zero();
                    }
                    sat_tile_offset =
                        sat_tile_offset_double.template cast<unsigned int>();

                    auto prev_slice_xy = new DoubleDataType[(m_tile_size + 1) *
                                                            (m_tile_size + 1)];
                    auto prev_slice_yz =
                        new DoubleDataType[(m_tile_size + 1) * m_tile_size];
                    auto prev_slice_xz =
                        new DoubleDataType[m_tile_size * m_tile_size];

                    DoubleDataType offset_to_substract =
                        sat_tile_offset.template cast<double>();
                    prev_slice_xy[0] =
                        sat_tile_offset_double - offset_to_substract;
                    if (tile_z > 0) {
                        auto &tile_tensor =
                            m_tiles_tensor(tile_z - 1, tile_y, tile_x);
                        for (Index y = 1; y < m_tile_size + 1; ++y) {
                            for (Index x = 1; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[y * (m_tile_size + 1) + x] =
                                    tile_tensor.get_value(x - 1, y - 1,
                                                          m_tile_size - 1) -
                                    offset_to_substract;
                            }
                        }
                        if (tile_y > 0) {
                            auto &tile_tensor2 =
                                m_tiles_tensor(tile_z - 1, tile_y - 1, tile_x);
                            for (Index x = 1; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[x] = tile_tensor2.get_value(
                                                       x - 1, m_tile_size - 1,
                                                       m_tile_size - 1) -
                                                   offset_to_substract;
                            }
                        } else {
                            for (Index x = 1; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[x] = DoubleDataType::Zero();
                            }
                        }
                        if (tile_x > 0) {
                            auto &tile_tensor3 =
                                m_tiles_tensor(tile_z - 1, tile_y, tile_x - 1);
                            for (Index y = 1; y < m_tile_size + 1; ++y) {
                                prev_slice_xy[y * (m_tile_size + 1)] =
                                    tile_tensor3.get_value(m_tile_size - 1,
                                                           y - 1,
                                                           m_tile_size - 1) -
                                    offset_to_substract;
                            }
                        } else {
                            for (Index y = 1; y < m_tile_size + 1; ++y) {
                                prev_slice_xy[y * (m_tile_size + 1)] =
                                    DoubleDataType::Zero();
                            }
                        }
                    } else {
                        for (Index y = 0; y < m_tile_size + 1; ++y) {
                            for (Index x = 0; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[y * (m_tile_size + 1) + x] =
                                    DoubleDataType::Zero();
                            }
                        }
                    }

                    if (tile_x > 0) {
                        auto &tile_tensor =
                            m_tiles_tensor(tile_z, tile_y, tile_x - 1);
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short y = 1; y < m_tile_size + 1;
                                 ++y) {
                                prev_slice_yz[z * (m_tile_size + 1) + y] =
                                    tile_tensor.get_value(m_tile_size - 1,
                                                          y - 1, z) -
                                    offset_to_substract;
                            }
                        }
                        if (tile_y > 0) {
                            auto &tile_tensor2 =
                                m_tiles_tensor(tile_z, tile_y - 1, tile_x - 1);
                            for (unsigned short z = 0; z < m_tile_size; ++z) {
                                prev_slice_yz[z * (m_tile_size + 1)] =
                                    tile_tensor2.get_value(m_tile_size - 1,
                                                           m_tile_size - 1, z) -
                                    offset_to_substract;
                            }
                        } else {
                            for (unsigned short z = 0; z < m_tile_size; ++z) {
                                prev_slice_yz[z * (m_tile_size + 1)] =
                                    DoubleDataType::Zero();
                            }
                        }
                    } else {
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short y = 0; y < m_tile_size + 1;
                                 ++y) {
                                prev_slice_yz[z * (m_tile_size + 1) + y] =
                                    DoubleDataType::Zero();
                            }
                        }
                    }

                    if (tile_y > 0) {
                        auto &tile_tensor =
                            m_tiles_tensor(tile_z, tile_y - 1, tile_x);
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short x = 0; x < m_tile_size; ++x) {
                                prev_slice_xz[z * m_tile_size + x] =
                                    tile_tensor.get_value(x, m_tile_size - 1,
                                                          z) -
                                    offset_to_substract;
                            }
                        }
                    } else {
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short x = 0; x < m_tile_size; ++x) {
                                prev_slice_xz[z * m_tile_size + x] =
                                    DoubleDataType::Zero();
                            }
                        }
                    }

                    if (all_values_equal_flag) {
                        delete[] tile_data;
                        m_tiles_tensor(tile_z, tile_y, tile_x)
                            .tile_set_data(m_tile_size, sat_tile_offset,
                                           prev_slice_xy, prev_slice_yz,
                                           prev_slice_xz, first_tile_value);
                    } else {
                        m_tiles_tensor(tile_z, tile_y, tile_x)
                            .tile_set_data(m_tile_size, sat_tile_offset,
                                           prev_slice_xy, prev_slice_yz,
                                           prev_slice_xz, tile_data);
                    }
                }
            }
        }
    }

    inline DoubleDataType get_sat_value(Index x, Index y, Index z) {
        if (x < 0 || y < 0 || z < 0) {
            return DoubleDataType::Zero();
        }
        return m_tiles_tensor(z / m_tile_size, y / m_tile_size, x / m_tile_size)
            .get_value(x - (x / m_tile_size) * m_tile_size,
                       y - (y / m_tile_size) * m_tile_size,
                       z - (z / m_tile_size) * m_tile_size);
    }

    inline DataType get_sum_from_sat(const BoundingBox &queryBox) {
        Index x1, x2, y1, y2, z1, z2;
        x1 = queryBox.lower[0] - 1;
        x2 = queryBox.upper[0] - 1;
        y1 = queryBox.lower[1] - 1;
        y2 = queryBox.upper[1] - 1;
        z1 = queryBox.lower[2] - 1;
        z2 = queryBox.upper[2] - 1;

        return (get_sat_value(x2, y2, z2) - get_sat_value(x1, y2, z2) +
                get_sat_value(x2, y1, z1) + get_sat_value(x1, y2, z1) -
                get_sat_value(x2, y1, z2) - get_sat_value(x2, y2, z1) +
                get_sat_value(x1, y1, z2) - get_sat_value(x1, y1, z1))
            .template cast<typename DataType::Scalar>();
    }
};

} // namespace TreeSAT

#ifdef SAT_TILE_TREE_BUILD_PYTHON
template <typename ValueType>
void bind_SATTileTree(py::module &m, std::string name) {
    using namespace TreeSAT;
    py::class_<SATTileTree<ValueType>>(m, name.c_str())
        .def(py::init<const py::array_t<float> &, unsigned short>(),
             py::arg("volume"), py::arg("tile_size") = 32)
        .def("query_average", &SATTileTree<ValueType>::queryAverageSlice)
        .def("query_singular", &SATTileTree<ValueType>::querySingularPy)
        .def("convert_dense", &SATTileTree<ValueType>::convertDense)
        .def("size", &SATTileTree<ValueType>::size)
        .def_property_readonly("shape", &SATTileTree<ValueType>::shape);
}

PYBIND11_MODULE(TARGET_NAME, m) {
    bind_SATTileTree<Eigen::Matrix<float, 1, 1>>(m, "SATTileTree");
    bind_SATTileTree<Eigen::Vector2d>(m, "SATTileTree2D");
}
#endif
