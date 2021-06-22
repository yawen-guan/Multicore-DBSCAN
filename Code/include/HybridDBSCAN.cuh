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

class HybridDBSCAN {
public:
    HybridDBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize, uint blockSize);
    void run();
    void print(const string &outFile);
    vector<uint> debug_getNeighbors(const uint &id);
    void debug_printNeighborTable();

    vector<int> clusterIDs;
    static const int UNVISITED = -1;
    static const int NOISE = -2;

private:
    void constructIndex();
    void calcCells();
    float constructGPUResultSet();
    void constructNeighborTable();
    void DBSCANwithNeighborTable();

    DataPointsType dataPoints;
    uint dataSize, gridSize, neighborsCnt;
    const float epsilon, epsilonPow;
    const uint minpts, blockSize;

    array<float, 2> minVals, maxVals;
    array<uint, 2> nCells;
    uint64 totalCells;
    // for each grid(aka. non-empty cell), orderedID in [orderedID_min, orderedID_max]
    vector<Grid> index;
    // orderedID 2 gridID
    vector<uint> ordered2GridID;
    // orderedID to dataID
    vector<uint> ordered2DataID;
    // dataID to orderedID
    vector<uint> data2OrderedID;
    // gridID to cellID
    vector<uint> grid2CellID;
    // cellID to gridID
    // unordered_map<uint, uint> cell2GridID;

    // gpuResultSet key, value
    uint *orderedIDKey, *orderedIDValue;

    // neighborTables[orderedID] = ...
    vector<NeighborTable> neighborTables;

    // uint *neighborTable;
    // vector<vector<uint>> neighborTable;
};