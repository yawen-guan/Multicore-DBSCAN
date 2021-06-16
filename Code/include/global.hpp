#pragma once

#include <array>
#include <string>
#include <tuple>
#include <vector>

using std::array;
using std::get;
using std::make_tuple;
using std::size_t;
using std::string;
using std::tie;
using std::tuple;
using std::vector;

#define NUM_THREADS 6

typedef unsigned int uint;
typedef unsigned long uint64;

typedef array<vector<float>, 2> DataPointsType;
typedef array<float, 2> MinMaxValsType;

struct PointChunkLookup {
    uint pointID;
    uint chunkID;
    uint idxInChunk;
};

struct Grid {
    uint indexMin;
    uint indexMax;
};

struct GridCellLookup {
    uint idx;                 // idx in the "grid" struct array
    uint64 gridCellLinearID;  // The linear ID of the grid cell
    bool operator<(const GridCellLookup& other) const {
        return gridCellLinearID < other.gridCellLinearID;
    }
};
