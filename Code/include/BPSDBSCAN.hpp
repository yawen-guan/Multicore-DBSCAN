#pragma once

#include "global.hpp"
#include "utils.hpp"

class BPSDBSCAN {
public:
    BPSDBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize, uint blockSize, const uint NCHUNKS);
    void run();
    void print(const string &outFile);
    vector<int> finalClusterIDs;
    static const int UNVISITED = -1;
    static const int NOISE = -2;

private:
    void partition();
    void modifiedDBSCAN();
    void merge();
    DataPointsType dataPoints;
    uint dataSize;
    const float epsilon, epsilonPow;
    const uint minpts, NCHUNKS, blockSize;

    array<float, 2> minVals, maxVals;
    array<uint, 2> nCells;
    uint64 totalCells;

    vector<vector<int>> clusterIDsArray;

    vector<uint> binBounaries;
    vector<PointChunkLookup> pointChunkMapping;
    vector<uint> pointIDs_shadow;
    array<vector<float>, 2> dataPoints_shadow;
    vector<DataPointsType> partDataPointsArray;
};