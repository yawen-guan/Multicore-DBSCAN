#include "utils.hpp"

DataPointsType importDataset(const string datafile) {
    return array<vector<float>, 2>();
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