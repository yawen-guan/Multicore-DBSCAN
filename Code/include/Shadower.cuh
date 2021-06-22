#pragma once

#include "global.hpp"
#include "HybridDBSCAN.cuh"
#include <unordered_map>


class Shadower {
public:
    Shadower(
        const float epsilon, 
        const uint minpts, 
        DataPointsType dataPoints, 
        uint dataSize, 
        uint blockSize, 
        vector<vector<int>> clusterIDsArray, 
        vector<uint> pointIDs_shadow, 
        vector<PointChunkLookup> pointChunkMapping);
    void run();
    
    unordered_map<int , int> merge;
    vector<int> clusterIDs;
    static const int UNVISITED = -1;
    static const int NOISE = -2;

private:
    void constructIndex();
    void calcCells();
    float constructGPUResultSet();
    void constructNeighborTable();
    void modifiedDBSCAN();

    DataPointsType dataPoints;
    uint dataSize, gridSize, neighborsCnt;
    const float epsilon, epsilonPow;
    const uint minpts, blockSize;

    array<float, 2> minVals, maxVals;
    array<uint, 2> nCells;
    uint64 totalCells;
    vector<Grid> index;
    vector<uint> ordered2GridID;
    vector<uint> ordered2DataID;
    vector<uint> data2OrderedID;
    vector<uint> grid2CellID;
    uint *orderedIDKey, *orderedIDValue;
    vector<NeighborTable> neighborTables;


    vector<vector<int>> clusterIDsArray;
    vector<uint> pointIDs_shadow;
    vector<PointChunkLookup> pointChunkMapping;

};