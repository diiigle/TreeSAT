#pragma once

namespace TreeSAT {
#define IMPORT_VALUETYPES(_DATATYPE)                                           \
    using DataType = _DATATYPE;                                                \
    enum {                                                                     \
        Dimensionality = BoundingBox::Scalar(DataType::RowsAtCompileTime),     \
    };                                                                         \
    using UIntDataType = Eigen::Matrix<unsigned int, Dimensionality, 1>;       \
    using DoubleDataType = Eigen::Matrix<double, Dimensionality, 1>;

// signed, compatibility with Eigen
// see
// https://eigen.tuxfamily.org/index.php?title=FAQ#Why_Eigen.27s_API_is_using_signed_integers_for_sizes.2C_indices.2C_etc..3F
using Index = long long int;

} // namespace TreeSAT