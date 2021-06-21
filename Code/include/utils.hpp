#pragma once

#include "global.hpp"

/**
 * @brief import input dataset
 */
DataPointsType importDataset(const string datafile);

/**
 * @param minVals minVal[i] = min(dataPoints[i] - epsilon)
 * @param maxVals maxVal[i] = max(dataPoints[i] + epsilon)
 * @param nCells nCells[i]: number of cells in ith dimension
 * @param totalCells 
 */
void generateGridDimensions(
    const DataPointsType &dataPoints,
    const float &epsilon,
    MinMaxValsType &minVals,
    MinMaxValsType &maxVals,
    array<uint, 2> &nCells,
    uint64 &totalCells);

/**
 * @param binBounaries bounaries for each bin (chunk)
 */
void generatePartitions(
    const DataPointsType &dataPoints,
    const float &epsilon,
    const MinMaxValsType &minVals,
    const MinMaxValsType &maxVals,
    const array<uint, 2> &nCells,
    const uint64 &totalCells,
    vector<uint> &binBounaries,
    const int &NCHUNKS);

/**
 * @brief generate dataset for each partition
 */
void generatePartitionDatasets(
    const DataPointsType &dataPoints,
    const float &epsilon,
    const MinMaxValsType &minVals,
    const MinMaxValsType &maxVals,
    const vector<uint> &binBounaries,
    const array<uint, 2> &nCells,
    vector<DataPointsType> &partDataPointsArray,
    vector<PointChunkLookup> &pointChunkMapping,
    vector<uint> &pointIDs_shadow,
    DataPointsType dataPoints_shadow);

void check(const uint dataSize, const vector<int> &clusterIDs0, const vector<int> &clusterIDs1);
