#include "BPSDBSCAN.hpp"

#include <fstream>
#include <iostream>

#include "HybridDBSCAN.hpp"
#include "Shadower.cuh"

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
    const uint NCHUNKS) : DBSCAN(epsilon, minpts, dataPoints, dataSize),
                          //   epsilon(epsilon),
                          //   epsilonPow(pow(epsilon, 2)),
                          //   minpts(minpts),
                          //   dataPoints(dataPoints),
                          //   dataSize(dataSize),
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

void BPSDBSCAN::merge() {
    unsigned long int sizeShadow = dataPoints_shadow[0].size();

    int nowindex = 0;
    int nextindex = 0;
    for (int i = 0; i < NCHUNKS; i++) {
        for (int j = 0; j < clusterIDsArray[i].size(); j++) {
            clusterIDsArray[i][j] += nowindex;
            if (clusterIDsArray[i][j] > nextindex) nextindex = clusterIDsArray[i][j];
        }
        nowindex = nextindex;
    }

    auto shadower = Shadower(
        epsilon,
        minpts,
        dataPoints_shadow,
        sizeShadow,
        blockSize,
        clusterIDsArray,
        pointIDs_shadow,
        pointChunkMapping);
    if (sizeShadow != 0) shadower.run();

    for (int i = 0; i < dataSize; i++) {
        int ind = clusterIDsArray[pointChunkMapping[i].chunkID][pointChunkMapping[i].idxInChunk];
        while (shadower.merge.find(ind) != shadower.merge.end()) {
            ind = shadower.merge[ind];
        }
        finalClusterIDs[i] = ind;
    }
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

        array<float, 2> partminVals, partmaxVals;
        array<uint, 2> partnCells;
        uint64 partnNonEmptyCells = 0, parttotalCells = 0, parttotalNeighbors = 0;
        generateGridDimensions(partDataPoints, epsilon, partminVals, partmaxVals, partnCells, parttotalCells);
        printf("Grid: total cells (including empty) %lu\n", totalCells);

        /**************** not finish *************/

        // Grid *index;
        GridCellLookup *gridCellLookup;
        // populateNDGridIndexAndLookupArrayParallel();

        /***** dynamic densebox *****/
        double meanPointsHeuristic = partDataSize / (8.0 * partnNonEmptyCells);
        if (meanPointsHeuristic < ((1.0 * minpts) * 0.25)) {
            enableDenseBox = false;
        }
        // printf("Dynamic densebox: Partition: %d, Heuristic value: %f, DENSEBOX: %d\n", partID, meanPointsHeuristic, enableDenseBox);

        vector<uint> queryPointIDs;  // Point ids that need to be searched on the GPU (weren't found by dense box)

        if (enableDenseBox) {
            auto denseBox = DenseBox(epsilon, minpts, partDataPoints);
            denseBox.test();
            partClusterIDs = denseBox.getCluster();
        } else {
            auto hybrid_dbscan = HybridDBSCAN(epsilon, minpts, partDataPoints, partDataPoints[0].size(), blockSize, 1);
            hybrid_dbscan.run();
            partClusterIDs.resize(hybrid_dbscan.clusterIDs.size());
            partClusterIDs = hybrid_dbscan.clusterIDs;
        }

        // #pragma omp critical
        {
            clusterIDsArray[partID] = partClusterIDs;
        }
    }
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
    auto start = system_clock::now();
    partition();
    modifiedDBSCAN();
    merge();
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Time elapsed on BPS-DBSCAN: "
         << double(duration.count()) * microseconds::period::num / milliseconds::period::den
         << "ms" << endl;
}