#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::stringstream;

DataPointsType importDataset(const string datafile) {
    DataPointsType dataPoints;
    ifstream in(datafile);
    string str;
    uint idx = 0;
    float x;

    if (in.is_open()) {
        while (getline(in, str, ',')) {
            stringstream ss(str);
            while (ss >> x) {
                dataPoints[idx].push_back(x);
                idx = 1 - idx;
            }
            // ;
            // cout << "str = " << str << " x = " << x << endl;
        }
        in.close();
    }

    return dataPoints;
}

void generateGridDimensions(
    const DataPointsType &dataPoints,
    const float &epsilon,
    MinMaxValsType &minVals,
    MinMaxValsType &maxVals,
    array<uint, 2> &nCells,
    uint64 &totalCells) {
}

void generatePartitions(
    const DataPointsType &dataPoints,
    const float &epsilon,
    const MinMaxValsType &minVals,
    const MinMaxValsType &maxVals,
    const array<uint, 2> &nCells,
    const uint64 &totalCells,
    vector<uint> &binBounaries,
    const int &NCHUNKS) {
}

void generatePartitionDatasets(
    const DataPointsType &dataPoints,
    const MinMaxValsType &minVals,
    const MinMaxValsType &maxVals,
    const vector<uint> &binBounaries,
    const array<uint, 2> &nCells,
    vector<DataPointsType> &partDataPointsArray,
    vector<PointChunkLookup> &pointChunkMapping,
    vector<uint> &pointIDs_shadow,
    DataPointsType dataPoints_shadow) {
}

void check(const uint dataSize, const vector<int> &clusterIDs0, const vector<int> &clusterIDs1) {
    for (int i = 0; i < dataSize; i++) {
        if (clusterIDs0[i] != clusterIDs1[i]) {
            fprintf(stderr, "Error: Wrong Answer.\n");
            return;
        }
    }
    printf("Check Passed.\n");
}