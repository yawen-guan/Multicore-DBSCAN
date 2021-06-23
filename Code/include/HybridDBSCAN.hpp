#pragma once

#include "DBSCAN.hpp"

class HybridDBSCAN : public DBSCAN {
public:
    HybridDBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize, uint blockSize, uint NCHUNKS);
    ~HybridDBSCAN();
    void run();
    void print(const string &outFile);
    vector<uint> debug_getNeighbors(const uint &id);
    void debug_printNeighborTable();

    vector<int> clusterIDs;

private:
    uint gridSize, neighborsCnts[GPU_STREAMS];  // neighborsCnt
    const uint blockSize, NCHUNKS;

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
    // uint *orderedIDKey, *orderedIDValue;
    uint *orderedIDKeys[GPU_STREAMS], *orderedIDValues[GPU_STREAMS];
    vector<uint *> valuePtrs;

    // neighborTables[orderedID] = ...
    vector<NeighborTable> neighborTables;
};