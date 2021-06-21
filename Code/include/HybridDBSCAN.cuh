#pragma once

#include "global.hpp"

struct Grid {
    uint orderedID_min;
    uint orderedID_max;
};

struct NeighborPair {
    uint orderedID_x;
    uint orderedID_y;
    NeighborPair(uint orderedID_x) : orderedID_x(orderedID_x) {}
    NeighborPair(uint orderedID_x, uint orderedID_y) : orderedID_x(orderedID_x), orderedID_y(orderedID_y) {}
};

class HybridDBSCAN {
public:
    HybridDBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize, uint blockSize);
    void run();
    void print(const string &outFile);
    void debug_printNeighborTable();

    vector<int> clusterIDs;
    static const int UNVISITED = -1;
    static const int NOISE = -2;

private:
    void constructIndex();
    void calcCells();
    float constructGPUResultSet();
    void modifiedDBSCAN();

    DataPointsType dataPoints;
    uint dataSize, gridSize;
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
    unordered_map<uint, uint> cell2GridID;
    // clusterIDs[dataID] = clusterID

    // uint *neighborTable;
    vector<vector<uint>> neighborTable;
};