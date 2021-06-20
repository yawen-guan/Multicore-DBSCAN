#pragma once

#include "global.hpp"

class DBSCAN {
public:
    DBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize);
    void run();
    void print(const string& outFile);

private:
    bool expandCluster(const uint& id, const uint& clusterID);
    float dist(const uint& id0, const uint& id1);
    vector<uint> getNeighbors(const uint& id);

    const float epsilon, epsilonPow;
    const uint minpts;
    DataPointsType dataPoints;
    uint dataSize;
    vector<uint> clusterIDs;

    static const uint UNVISITED = -1;
    static const uint NOISE = -2;
};
