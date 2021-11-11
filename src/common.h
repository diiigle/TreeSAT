#pragma once

#define IMPORT_VALUETYPES(_DATATYPE) using DataType = _DATATYPE;\
    enum {\
        Dimensionality = BoundingBox::Scalar(DataType::RowsAtCompileTime),\
    };\
    using UIntDataType = Eigen::Matrix<unsigned int, Dimensionality, 1>;\
    using DoubleDataType = Eigen::Matrix<double, Dimensionality, 1>;
