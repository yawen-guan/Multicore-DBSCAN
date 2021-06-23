#pragma once

#include <chrono>

#include "global.hpp"
using namespace std::chrono;

struct Grid {
    uint orderedID_min;
    uint orderedID_max;
};

struct NeighborTable {
    int valueIdx_min;
    int valueIdx_max;
    uint *values;
    NeighborTable() : valueIdx_min(-1), valueIdx_max(-1), values(nullptr) {}
};

struct KeyData {
    uint key, pos;
    KeyData(uint key, uint pos) : key(key), pos(pos) {}
};

class DBSCAN {
public:
    DBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize);
    ~DBSCAN();
    virtual void run() = 0;
    virtual void print(const string &outFile) = 0;

    static const int UNVISITED = -1;
    static const int NOISE = -2;

    void test();

protected:
    void calcCells(
        array<float, 2> &minVals,
        array<float, 2> &maxVals,
        array<uint, 2> &nCells,
        uint64 &totalCells);

    void constructIndex(
        const array<float, 2> &minVals,
        const array<float, 2> &maxVals,
        const array<uint, 2> &nCells,
        const uint64 &totalCells,
        uint &gridSize,
        vector<Grid> &index,
        vector<uint> &ordered2GridID,
        vector<uint> &ordered2DataID,
        vector<uint> &data2OrderedID,
        vector<uint> &grid2CellID);

    float constructGPUResultSet(
        const uint &chunkID,
        const uint &NCHUNKS,
        const array<uint, 2> &nCells,
        const uint &gridSize,
        const vector<uint> &ordered2GridID,
        const vector<uint> &ordered2DataID,
        const vector<uint> &grid2CellID,
        const vector<Grid> &index,
        const uint &blockSize,
        uint &neighborsCnt,
        uint *&orderedIDKey,
        uint *&orderedIDValue);

    void constructNeighborTable(
        const uint *orderedIDKey,
        const uint *orderedIDValue,
        const uint &neighborsCnt,
        uint *&valuePtr,
        vector<NeighborTable> &neighborTables);

    void DBSCANwithNeighborTable(
        const vector<uint> &data2OrderedID,
        const vector<uint> &ordered2DataID,
        const vector<NeighborTable> &neighborTables,
        vector<int> &clusterIDs);

    void constructResultSetAndNeighborTable(
        const uint &NCHUNKS,
        const array<uint, 2> &nCells,
        const uint &gridSize,
        const vector<uint> &ordered2GridID,
        const vector<uint> &ordered2DataID,
        const vector<uint> &grid2CellID,
        const vector<Grid> &index,
        const uint &blockSize,
        uint neighborsCnts[GPU_STREAMS],
        uint *orderedIDKeys[GPU_STREAMS],
        uint *orderedIDValue[GPU_STREAMS],
        vector<uint *> &valuePtrs,
        vector<NeighborTable> &neighborTables);

    const float epsilon, epsilonPow;
    const uint minpts;
    DataPointsType dataPoints;
    uint dataSize;
};
