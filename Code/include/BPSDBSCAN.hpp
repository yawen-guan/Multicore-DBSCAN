#pragma once

#include "DBSCAN.hpp"
#include "DenseBox.hpp"
#include "global.hpp"
#include "utils.hpp"

class BPSDBSCAN : public DBSCAN {
public:
    BPSDBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize, uint blockSize, const uint NCHUNKS);
    void run();
    void print(const string &outFile);
    vector<int> finalClusterIDs;

private:
    void partition();
    void modifiedDBSCAN();
    void merge();

    const uint NCHUNKS, blockSize;

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