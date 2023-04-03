// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "in_mem_data_store.h"

#include "utils.h"

namespace diskann
{

template <typename data_t>
InMemDataStore<data_t>::InMemDataStore(const location_t max_pts, 
                                             const size_t dim, std::shared_ptr<Distance<data_t>> distance_metric)
    : AbstractDataStore<data_t>(max_pts, dim), _aligned_dim(ROUND_UP(dim, 8)), 
      _distance_metric(distance_metric)
{
    alloc_aligned(((void **)&_data), max_pts * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    std::memset(_data, 0, max_pts * _aligned_dim * sizeof(data_t));
}

template <typename data_t> InMemDataStore<data_t>::~InMemDataStore()
{
    if (_data != nullptr)
    {
        aligned_free(this->_data);
    }
}

template <typename data_t> void InMemDataStore<data_t>::load(const std::string &filename)
{
    load_data(filename);
}

template <typename data_t> void InMemDataStore<data_t>::store(const std::string &filename)
{
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t Index<data_t>::load_data(AlignedFileReader &reader)
{
    size_t file_dim, file_num_points;

    diskann::get_bin_metadata(reader, file_num_points, file_dim);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > _max_points + _num_frozen_pts)
    {
        resize(file_num_points - _num_frozen_pts);
    }

    return file_num_points;
}
#endif

template <typename data_t>
location_t InMemDataStore<data_t>::load_data(const std::string &filename)
{
    size_t file_dim, file_num_points;
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->get_max_points())
    {
        resize(file_num_points);
    }

    copy_aligned_data_from_file<data_t>(filename.c_str(), _data, file_num_points, file_dim, _aligned_dim);

    return file_num_points;
}

template <typename data_t> data_t *InMemDataStore<data_t>::get_vector(location_t i)
{
    return _data + i * _aligned_dim;
}

template <typename data_t>
void InMemDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    memcpy(_data + loc * _aligned_dim, vector, this->_dim * sizeof(data_t));
}

} // namespace diskann