#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <map>
#include <set>

#include "global.hpp"
using std::initializer_list;
using std::map;
using std::max;
using std::min;
using std::set;
using std::pair;

struct Point {
    float x, y;
    Point(const float& x, const float& y) : x(x), y(y){};
    const bool operator<(const Point& other) const {
        if (x < other.x)
            return true;
        return y < other.y;
    }
};

class DenseDBSCAN {
public:
    DenseDBSCAN(const float epsilon, const uint minpts, DataPointsType dataPoints);
    void run();
    vector<int>& getCluster();

private:
    bool expandCluster(const uint& id, const uint& clusterID);
    float dist(const uint& id0, const uint& id1);
    vector<uint> getNeighbors(const uint& id);

    const float epsilon, epsilonPow;
    const uint minpts;
    DataPointsType dataPoints;
    uint dataSize;
    vector<int> clusterIDs;
    static const uint UNVISITED = -1;
    static const uint NOISE = -2;
};

class DenseBox {
public:
    DenseBox(const float epsilon,
             const uint minpts,
             DataPointsType dataPoints);
    void run();
    void test();
    vector<int> getCluster();

private:
    float _epsilon,_oriEpsilon;
    uint _minpts;
    array<float, 2> _minVals, _maxVals;
    DataPointsType _dataPoints;
    uint _dataSize;
    array<int, 2> _nCells;
    uint64 _totalCells;
    vector<vector<uint>> _cellInners;
    vector<uint> _nonEmptyCellIDs;
    map<Point, int> _point2ID;
    vector<int> _DBCluster, _DBSCANCluster;
    DataPointsType _densePoints;
    vector<int> _densePointsID2dataPointsID;
    uint _curClusterNums;

    void calcCells();
    void clusterAndGetDensePoints();
    void genReferencePoints();
    void fastDBSCAN();
    void reInterpretCluster();
    void reArrangeCluster();
    void mergeAdjacent();
    vector<int> getCellFromStepList(int& cellID, vector<int>& stepList);
    bool needMerge(int& cellID1,int& cellID2);
    void mergeNonAdjacent();
    void mergeAll();
    float dist(const uint& id0, const uint& id1);
};