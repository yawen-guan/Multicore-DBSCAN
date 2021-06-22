#pragma once

#include <omp.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using std::array;
using std::get;
using std::make_tuple;
using std::pow;
using std::size_t;
using std::string;
using std::tie;
using std::tuple;
using std::unordered_map;
using std::unordered_set;
using std::vector;

#define NUM_THREADS 6
#define GPU_BUFFER_SIZE 100000000

typedef unsigned int uint;
typedef unsigned long uint64;

typedef array<vector<float>, 2> DataPointsType;
typedef array<float, 2> MinMaxValsType;

struct PointChunkLookup {
    uint pointID;
    uint chunkID;
    uint idxInChunk;
};

struct GridCellLookup {
    uint idx;                 // idx in the "grid" struct array
    uint64 gridCellLinearID;  // The linear ID of the grid cell
    bool operator<(const GridCellLookup& other) const {
        return gridCellLinearID < other.gridCellLinearID;
    }
};

//need to pass in the neighbortable thats an array of the dataset size.
//carry around a pointer to the array that has the points within epsilon though
struct NeighborTableLookup {
    uint pointID;
    uint indexMin;
    uint indexMax;
    int* dataPtr;
};
