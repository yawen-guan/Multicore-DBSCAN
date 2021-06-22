#pragma once

#include "global.hpp"

struct Grid {
    uint orderedID_min;
    uint orderedID_max;
};

struct NeighborTable {
    uint valueIdx_min;
    uint valueIdx_max;
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

protected:
    void calcCells(
        array<float, 2> &minVals,
        array<float, 2> &maxVals,
        array<uint, 2> &nCells,
        uint64 &totalCells);

    void constructIndex(
        array<float, 2> &minVals,
        array<float, 2> &maxVals,
        array<uint, 2> &nCells,
        uint64 &totalCells,
        uint &gridSize,
        vector<Grid> &index,
        vector<uint> &ordered2GridID,
        vector<uint> &ordered2DataID,
        vector<uint> &data2OrderedID,
        vector<uint> &grid2CellID);

    float constructGPUResultSet(
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
        vector<NeighborTable> &neighborTables);

    void DBSCANwithNeighborTable(
        const vector<uint> &data2OrderedID,
        const vector<uint> &ordered2DataID,
        const uint *orderedIDValue,
        const vector<NeighborTable> &neighborTables,
        vector<int> &clusterIDs);

    const float epsilon, epsilonPow;
    const uint minpts;
    DataPointsType dataPoints;
    uint dataSize;
};
