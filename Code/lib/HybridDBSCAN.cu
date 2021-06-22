#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fstream>
#include <iostream>
#include <parallel/algorithm>

#include "HybridDBSCAN.cuh"

using std::ceil;
using std::cout;
using std::endl;
using std::iostream;
using std::lower_bound;
using std::max;
using std::min;
using std::ofstream;
using thrust::device_vector;

HybridDBSCAN::HybridDBSCAN(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints,
    uint dataSize,
    uint blockSize) : DBSCAN(epsilon, minpts, dataPoints, dataSize),
                      blockSize(blockSize),
                      gridSize(0) {
    clusterIDs.resize(dataSize);
    for (int i = 0; i < dataSize; i++) {
        clusterIDs[i] = HybridDBSCAN::UNVISITED;
    }
}

void HybridDBSCAN::run() {
    float start = omp_get_wtime();
    constructIndex(
        minVals,
        maxVals,
        nCells,
        totalCells,
        gridSize,
        index,
        ordered2GridID,
        ordered2DataID,
        data2OrderedID,
        grid2CellID);
    float gpu_elapsed_time_ms = constructGPUResultSet(
        nCells,
        gridSize,
        ordered2GridID,
        ordered2DataID,
        grid2CellID,
        index,
        blockSize,
        neighborsCnt,
        orderedIDKey,
        orderedIDValue);
    constructNeighborTable(
        orderedIDKey,
        orderedIDValue,
        neighborsCnt,
        neighborTables);
    DBSCANwithNeighborTable(
        data2OrderedID,
        ordered2DataID,
        orderedIDValue,
        neighborTables,
        clusterIDs);
    float end = omp_get_wtime();
    float elapsed_time_ms = (end - start) * 1000;
    printf("Time elapsed on Hybrid-DBSCAN: %f ms; gpu_elapsed_time: %f ms\n", elapsed_time_ms, gpu_elapsed_time_ms);
}

void HybridDBSCAN::print(const string &outFile) {
    ofstream out(outFile);
    if (out.is_open()) {
        out << "x,y,clusterID\n";
        char buffer[200];
        for (uint i = 0; i < dataSize; i++) {
            sprintf(buffer, "%5.15lf,%5.15lf,%d\n", dataPoints[0][i], dataPoints[1][i], clusterIDs[i]);
            out << buffer;
        }
        out.close();
    } else {
        cout << "Unable to open file " << outFile << endl;
    }
}

vector<uint> HybridDBSCAN::debug_getNeighbors(const uint &id) {
    uint orderedID = data2OrderedID[id];
    auto neighbors = vector<uint>();
    for (int i = neighborTables[orderedID].valueIdx_min; i <= neighborTables[orderedID].valueIdx_max; i++) {
        neighbors.push_back(ordered2DataID[orderedIDValue[i]]);
    }
    sort(neighbors.begin(), neighbors.end());
    return neighbors;
}

void HybridDBSCAN::debug_printNeighborTable() {
    for (int i = 0; i < dataSize; i++) {
        uint orderedID = data2OrderedID[i];
        printf("\ndataID = %d, neighbors's dataID: ", i);
        for (int j = neighborTables[orderedID].valueIdx_min; j <= neighborTables[orderedID].valueIdx_max; j++) {
            printf("%d, ", ordered2DataID[orderedIDValue[j]]);
        }
    }
    printf("\n");
}