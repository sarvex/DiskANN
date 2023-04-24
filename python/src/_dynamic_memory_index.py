# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import warnings
from typing import Literal, Optional, Tuple

import numpy as np

from . import _diskannpy as _native_dap
from ._common import (
    VectorDType,
    _assert,
    _assert_2d,
    _assert_dtype,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _get_valid_metric,
)
from ._diskannpy import defaults

__ALL__ = ["DynamicMemoryIndex"]


class DynamicMemoryIndex:
    def __init__(
        self,
        metric: Literal["l2", "mips"],
        vector_dtype: VectorDType,
        dim: int,
        max_points: int,
        complexity: int,
        graph_degree: int,
        saturate_graph: bool = defaults.SATURATE_GRAPH,
        max_occlusion_size: int = defaults.MAX_OCCLUSION_SIZE,
        alpha: float = defaults.ALPHA,
        num_threads: int = defaults.NUM_THREADS,
        filter_complexity: int = defaults.FILTER_COMPLEXITY,
        num_frozen_points: int = defaults.NUM_FROZEN_POINTS_DYNAMIC,
        initial_search_complexity: int = 0,
        search_threads: int = 0,
        concurrent_consolidation: bool = True,
        index_path: Optional[str] = None,
    ):
        """
        The diskannpy.DynamicMemoryIndex represents our python API into a dynamic DiskANN InMemory Index library.

        This dynamic index is unlike the DiskIndex and StaticMemoryIndex, in that after loading it you can continue
        to insert and delete vectors.

        Deletions are completed lazily, until the user executes `DynamicMemoryIndex.consolidate_deletes()`

        :param metric: One of {"l2", "mips"}. L2 is supported for all 3 vector dtypes, but MIPS is only
            available for single point floating numbers (numpy.single)
        :type metric: str
        :param vector_dtype: The vector dtype this index will be exposing.
        :type vector_dtype: Type[numpy.single], Type[numpy.byte], Type[numpy.ubyte]
        :param dim: The vector dimensionality of this index. All new vectors inserted must be the same dimensionality.
        :type dim: int
        :param max_points: Capacity of the data store for future insertions
        :type max_points: int
        :param graph_degree: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
            result in larger indices and longer indexing times, but better search quality.
        :type graph_degree: int
        :param saturate_graph:
        :type saturate_graph: bool
        :param max_occlusion_size:
        :type max_occlusion_size: int
        :param alpha:
        :type alpha: float
        :param num_threads:
        :type num_threads: int
        :param filter_complexity:
        :type filter_complexity: int
        :param num_frozen_points:
        :type num_frozen_points: int
        :param initial_search_complexity: The working scratch memory allocated is predicated off of
            initial_search_complexity * search_threads. If a larger list_size * num_threads value is
            ultimately provided by the individual action executed in `batch_query` than provided in this constructor,
            the scratch space is extended. If a smaller list_size * num_threads is provided by the action than the
            constructor, the pre-allocated scratch space is used as-is.
        :type initial_search_complexity: int
        :param search_threads: Should be set to the most common batch_query num_threads size. The working
            scratch memory allocated is predicated off of initial_search_list_size * initial_search_threads. If a
            larger list_size * num_threads value is ultimately provided by the individual action executed in
            `batch_query` than provided in this constructor, the scratch space is extended. If a smaller
            list_size * num_threads is provided by the action than the constructor, the pre-allocated scratch space
            is used as-is.
        :type search_threads: int
        :param concurrent_consolidation:
        :type concurrent_consolidation: bool
        :param index_path: Path on disk where the disk index is stored. Default is `None`.
        :type index_path: Optional[str]
        """
        dap_metric = _get_valid_metric(metric)
        _assert_dtype(vector_dtype, "vector_dtype")
        self._vector_dtype = vector_dtype

        _assert_is_positive_uint32(dim, "dim")
        _assert_is_positive_uint32(max_points, "max_points")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_positive_uint32(graph_degree, "graph_degree")
        _assert_is_nonnegative_uint32(max_occlusion_size, "max_occlusion_size")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
        _assert_is_nonnegative_uint32(num_frozen_points, "num_frozen_points")
        _assert_is_nonnegative_uint32(
            initial_search_complexity, "initial_search_complexity"
        )
        _assert_is_nonnegative_uint32(search_threads, "search_threads")

        self._dims = dim
        self._index_path = index_path if index_path is not None else ""

        if vector_dtype == np.single:
            _index = _native_dap.DynamicMemoryFloatIndex
        elif vector_dtype == np.ubyte:
            _index = _native_dap.DynamicMemoryUInt8Index
        else:
            _index = _native_dap.DynamicMemoryInt8Index
        self._index = _index(
            metric=dap_metric,
            dim=dim,
            max_points=max_points,
            complexity=complexity,
            graph_degree=graph_degree,
            saturate_graph=saturate_graph,
            max_occlusion_size=max_occlusion_size,
            alpha=alpha,
            num_threads=num_threads,
            filter_complexity=filter_complexity,
            num_frozen_points=num_frozen_points,
            initial_search_complexity=initial_search_complexity,
            search_threads=search_threads,
            concurrent_consolidation=concurrent_consolidation,
            index_path=self._index_path,
        )

    def search(
        self, query: np.ndarray, k_neighbors: int, complexity: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the disk index by a single query vector in a 1d numpy array.

        numpy array dtype must match index.

        :param query: 1d numpy array of the same dimensionality and dtype of the index.
        :type query: numpy.ndarray
        :param k_neighbors: Number of neighbors to be returned. If query vector exists in index, it almost definitely
            will be returned as well, so adjust your ``k_neighbors`` as appropriate. (> 0)
        :type k_neighbors: int
        :param complexity: Size of list to use while searching. List size increases accuracy at the cost of latency. Must
            be at least k_neighbors in size.
        :type complexity: int
        :return: Returns a tuple of 1-d numpy ndarrays; the first including the indices of the approximate nearest
            neighbors, the second their distances. These are aligned arrays.
        """
        _assert(len(query.shape) == 1, "query vector must be 1-d")
        _assert(
            query.dtype == self._vector_dtype,
            f"DynamicMemoryIndex was built expecting a dtype of {self._vector_dtype}, but the query vector is of dtype "
            f"{query.dtype}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_nonnegative_uint32(complexity, "complexity")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors
        return self._index.search(query=query, knn=k_neighbors, complexity=complexity)

    def batch_search(
        self, queries: np.ndarray, k_neighbors: int, complexity: int, num_threads: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the disk index for many query vectors in a 2d numpy array.

        numpy array dtype must match index.

        This search is parallelized and far more efficient than searching for each vector individually.

        :param queries: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
            number of queries intended to search for in parallel. Dtype must match dtype of the index.
        :type queries: numpy.ndarray
        :param k_neighbors: Number of neighbors to be returned. If query vector exists in index, it almost definitely
            will be returned as well, so adjust your ``k_neighbors`` as appropriate. (> 0)
        :type k_neighbors: int
        :param complexity: Size of list to use while searching. List size increases accuracy at the cost of latency. Must
            be at least k_neighbors in size.
        :type complexity: int
        :param num_threads: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        :type num_threads: int
        :return: Returns a tuple of 2-d numpy ndarrays; each row corresponds to the query vector in the same index,
            and elements in row corresponding from 1..k_neighbors approximate nearest neighbors. The second ndarray
            contains the distances, of the same form: row index will match query index, column index refers to
            1..k_neighbors distance. These are aligned arrays.
        """
        _assert_2d(queries, "queries")
        _assert(
            queries.dtype == self._vector_dtype,
            f"StaticMemoryIndex was built expecting a dtype of {self._vector_dtype}, but the query vectors are of dtype "
            f"{queries.dtype}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        num_queries, dim = queries.shape
        return self._index.batch_search(
            queries=queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            num_threads=num_threads,
        )

    def save(self, save_path: str = "", compact_before_save: bool = False):
        if save_path == "" and self._index_path == "":
            raise ValueError(
                "save_path cannot be empty if index_path is not set to a valid path in the constructor"
            )
        self._index.save(save_path=save_path, compact_before_save=compact_before_save)

    def insert(self, vector: np.ndarray, vector_id: int):
        _assert(len(vector.shape) == 1, "insert vector must be 1-d")
        _assert(
            vector.dtype == self._vector_dtype,
            f"DynamicMemoryIndex was built expecting a dtype of {self._vector_dtype}, but the insert vector is of dtype "
            f"{vector.dtype}",
        )
        _assert_is_positive_uint32(vector_id, "vector_id")
        return self._index.insert(vector, vector_id)

    def batch_insert(
        self,
        vectors: np.ndarray,
        vector_ids: np.ndarray,
        num_threads: int = 0
    ):
        """       
        :param num_threads: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        :type num_threads: int
        """
        _assert(len(vectors.shape) == 2, "vectors must be a 2-d array")
        _assert(
            vectors.dtype == self._vector_dtype,
            f"DynamicMemoryIndex was built expecting a dtype of {self._vector_dtype}, but the insert vector is of dtype "
            f"{vectors.dtype}",
        )
        _assert(vectors.shape[0] == vector_ids.shape[0], "#vectors must be equal to #ids")
        # Add a check on ID values
        return self._index.batch_insert(vectors, vector_ids, vector_ids.shape[0], num_threads)

    def mark_deleted(self, vector_id: int):
        _assert_is_positive_uint32(vector_id, "vector_id")
        self._index.mark_deleted(vector_id)

    def consolidate_delete(self):
        self._index.consolidate_delete()