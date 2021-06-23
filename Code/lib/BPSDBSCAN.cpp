#include "BPSDBSCAN.hpp"

#include <fstream>
#include <iostream>

#include "HybridDBSCAN.hpp"

using std::ceil;
using std::cout;
using std::endl;
using std::iostream;
using std::lower_bound;
using std::max;
using std::min;
using std::ofstream;

BPSDBSCAN::BPSDBSCAN(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints,
    uint dataSize,
    uint blockSize,
    const uint NCHUNKS) : epsilon(epsilon),
                          epsilonPow(pow(epsilon, 2)),
                          minpts(minpts),
                          dataPoints(dataPoints),
                          dataSize(dataSize),
                          blockSize(blockSize),
                          NCHUNKS(NCHUNKS) {
    finalClusterIDs.resize(dataSize);

    for (int i = 0; i < dataSize; i++) {
        finalClusterIDs[i] = BPSDBSCAN::UNVISITED;
    }
    clusterIDsArray.resize(NCHUNKS);
    for (int i = 0; i < NCHUNKS; i++) {
        clusterIDsArray[i].clear();
        clusterIDsArray[i].shrink_to_fit();
    }
    binBounaries.resize(NCHUNKS + 1);
}

void BPSDBSCAN::partition() {
    generateGridDimensions(dataPoints, epsilon, minVals, maxVals, nCells, totalCells);

    generatePartitions(
        dataPoints,
        epsilon,
        minVals,
        maxVals,
        nCells,
        totalCells,
        binBounaries,
        NCHUNKS);

    generatePartitionDatasets(
        dataPoints,
        epsilon,
        minVals,
        maxVals,
        binBounaries,
        nCells,
        partDataPointsArray,
        pointChunkMapping,
        pointIDs_shadow,
        dataPoints_shadow);
}

void BPSDBSCAN::modifiedDBSCAN() {
    bool enableDenseBox = true;

#pragma omp parallel for num_threads(NUM_THREADS) shared(clusterIDsArray, partDataPointsArray) schedule(dynamic, 1)
    for (uint partID = 0; partID < NCHUNKS; partID++) {
        int tid = omp_get_thread_num();

        DataPointsType partDataPoints = partDataPointsArray[partID];
        uint64 partDataSize = partDataPoints[0].size();
        printf("Data points in partition #%d: %lu\n", partID, partDataSize);

        vector<int> partClusterIDs;  // local cluster IDs

        array<float, 2> minVals, maxVals;
        array<uint, 2> nCells;
        uint64 nNonEmptyCells = 0, totalCells = 0, totalNeighbors = 0;
        generateGridDimensions(partDataPoints, epsilon, minVals, maxVals, nCells, totalCells);
        printf("Grid: total cells (including empty) %lu\n", totalCells);

        /**************** not finish *************/

        // Grid *index;
        GridCellLookup *gridCellLookup;
        // populateNDGridIndexAndLookupArrayParallel();

        /***** dynamic densebox *****/
        double meanPointsHeuristic = partDataSize / (8.0 * nNonEmptyCells);
        if (meanPointsHeuristic < ((1.0 * minpts) * 0.25)) {
            enableDenseBox = false;
        }
        // printf("Dynamic densebox: Partition: %d, Heuristic value: %f, DENSEBOX: %d\n", partID, meanPointsHeuristic, enableDenseBox);

        vector<uint> queryPointIDs;  // Point ids that need to be searched on the GPU (weren't found by dense box)

        if (enableDenseBox && 0) {
            /***** handle densebox *****/
        } else {
            auto hybrid_dbscan = HybridDBSCAN(epsilon, minpts, partDataPoints, partDataPoints[0].size(), blockSize, 1);
            hybrid_dbscan.run();
            partClusterIDs.resize(hybrid_dbscan.clusterIDs.size());
            partClusterIDs = hybrid_dbscan.clusterIDs;
        }

#pragma omp critical
        {
            clusterIDsArray[partID] = partClusterIDs;
        }
    }
}

void BPSDBSCAN::merge() {
    finalClusterIDs = clusterIDsArray[0];
}

void BPSDBSCAN::print(const string &outFile) {
    ofstream out(outFile);
    if (out.is_open()) {
        out << "x,y,clusterID\n";
        char buffer[200];
        for (uint i = 0; i < dataSize; i++) {
            sprintf(buffer, "%5.15lf,%5.15lf,%d\n", dataPoints[0][i], dataPoints[1][i], finalClusterIDs[i]);
            out << buffer;
        }
        out.close();
    } else {
        cout << "Unable to open file " << outFile << endl;
    }
}

void BPSDBSCAN::run() {
    float start = omp_get_wtime();
    partition();
    modifiedDBSCAN();
    merge();
    float end = omp_get_wtime();
    float elapsed_time_ms = (end - start) * 1000;
    printf("Time elapsed on BPS-DBSCAN: %f ms\n", elapsed_time_ms);
}