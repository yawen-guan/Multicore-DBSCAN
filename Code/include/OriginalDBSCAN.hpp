#pragma once

#include "DBSCAN.hpp"

class OriginalDBSCAN : public DBSCAN {
public:
    OriginalDBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints, uint dataSize);
    void run();
    void print(const string& outFile);
    void debug_printNeighborTable();
    vector<uint> getNeighbors(const uint& id);

    vector<int> clusterIDs;
    static const uint UNVISITED = -1;
    static const uint NOISE = -2;

private:
    bool expandCluster(const uint& id, const uint& clusterID);
    float dist(const uint& id0, const uint& id1);
};
