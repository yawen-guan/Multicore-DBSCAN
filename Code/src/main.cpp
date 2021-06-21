#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>

#include "DBSCAN.hpp"
#include "HybridDBSCAN.cuh"
#include "global.hpp"
#include "utils.hpp"

using std::cin;
using std::cout;
using std::endl;

void BPS_DBSCAN(
    const float epsilon,
    const uint minpts,
    const uint NCHUNKS,
    vector<vector<uint>> clusterIDsArray,
    vector<DataPointsType> partDataPointsArray) {
    bool enableDenseBox = true;
    double startTime_DBSCAN = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THREADS) shared(clusterIDsArray, partDataPointsArray) schedule(dynamic, 1)
    for (uint partID = 0; partID < NCHUNKS; partID++) {
        int tid = omp_get_thread_num();

        DataPointsType partDataPoints = partDataPointsArray[partID];
        uint64 partDataSize = partDataPoints[0].size();
        printf("Data points in partition #%d: %lu", partID, partDataSize);

        vector<uint> partClusterIDs;  // local cluster IDs

        array<float, 2> minVals, maxVals;
        array<uint, 2> nCells;
        uint64 nNonEmptyCells = 0, totalCells = 0, totalNeighbors = 0;
        generateGridDimensions(partDataPoints, epsilon, minVals, maxVals, nCells, totalCells);
        printf("Grid: total cells (including empty) %lu", totalCells);

        /**************** not finish *************/

        Grid *index;
        GridCellLookup *gridCellLookup;
        // populateNDGridIndexAndLookupArrayParallel();

        /***** dynamic densebox *****/
        double meanPointsHeuristic = partDataSize / (8.0 * nNonEmptyCells);
        if (meanPointsHeuristic < ((1.0 * minpts) * 0.25)) {
            enableDenseBox = false;
        }
        printf("Dynamic densebox: Partition: %d, Heuristic value: %f, DENSEBOX: %d", partID, meanPointsHeuristic, enableDenseBox);

        vector<uint> queryPointIDs;  // Point ids that need to be searched on the GPU (weren't found by dense box)

        if (enableDenseBox) {
            /***** handle densebox *****/
        } else {
            /***** HYBRID-DBSCAN *****/
        }

#pragma omp critical
        {
            clusterIDsArray[partID] = partClusterIDs;
        }
    }

    double endTime_DBSCAN = omp_get_wtime();
    printf("\n----------- BPS-HDBSCAN -----------\n");
    printf("Time total to BPS-HDBSCAN: %f\n", endTime_DBSCAN - startTime_DBSCAN);
}

int main(int argc, char *argv[]) {
    /***** handle input *****/

    if (argc < 6) {
        fprintf(stderr, "Usage: multicoreDBSCAN <Dataset file>, <Epsilon>, <Minpts>, <Partitions>, <BlockSize>");
        exit(0);
    }

    const string datafile = string(argv[1]);
    const float epsilon = atof(argv[2]);
    const uint minpts = atoi(argv[3]);
    const uint NCHUNKS = atoi(argv[4]);
    const uint blockSize = atoi(argv[5]);

    printf("\n----------- handle input -----------\n");
    cout << "datafile = " << datafile << ", epsilon = " << epsilon << ", minpts = " << minpts << ", NCHUNK = " << NCHUNKS << endl;

    /***** import dataset *****/

    array<vector<float>, 2> dataPoints = importDataset(datafile);
    uint dataSize = dataPoints[0].size();

    printf("\n----------- import dataset -----------\n");
    printf("dataSize = %u\n", dataSize);

    /***** original DBSCAN *****/

    auto dbscan = DBSCAN(epsilon, minpts, dataPoints, dataSize);
    dbscan.run();
    printf("\n----------- original DBSCAN -----------\n");
    dbscan.print("../data/output/DBSCAN-data-2500-out.csv");

    /***** Hybrid DBSCAN *****/

    auto hybrid_dbscan = HybridDBSCAN(epsilon, minpts, dataPoints, dataSize, blockSize);
    hybrid_dbscan.run();
    printf("\n----------- Hybrid DBSCAN -----------\n");
    hybrid_dbscan.print("../data/output/Hybrid-DBSCAN-data-2500-out.csv");

    check(dataSize, dbscan.clusterIDs, hybrid_dbscan.clusterIDs);

    /***** initial clusters *****/

    vector<vector<uint>> clusterIDsArray(NCHUNKS);  // one set of cluster IDs per chunk
    for (int i = 0; i < NCHUNKS; i++) {
        clusterIDsArray[i].clear();
        clusterIDsArray[i].shrink_to_fit();
    }
    vector<uint> finalClusterIDs;  // final set of cluster IDs when all chunk are merged
    finalClusterIDs.clear();
    finalClusterIDs.shrink_to_fit();

    printf("\n----------- initial clusters -----------\n");

    /***** generate grid dimensions *****/

    array<float, 2> minVals, maxVals;
    array<uint, 2> nCells;
    uint64 totalCells = 0;
    generateGridDimensions(dataPoints, epsilon, minVals, maxVals, nCells, totalCells);

    printf("\n----------- generate grid dimenions -----------\n");
    printf("Grid: total cells (including empty) %ld\n", totalCells);

    /***** generate partition & partition datasets *****/

    vector<uint> binBounaries(NCHUNKS + 1);
    vector<PointChunkLookup> pointChunkMapping;
    vector<uint> pointIDs_shadow;
    array<vector<float>, 2> dataPoints_shadow;
    vector<DataPointsType> partDataPointsArray;

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
        minVals,
        maxVals,
        binBounaries,
        nCells,
        partDataPointsArray,
        pointChunkMapping,
        pointIDs_shadow,
        dataPoints_shadow);

    printf("\n----------- generate partition & partition datasets -----------\n");

    /***** BPS-DBSCAN *****/

    BPS_DBSCAN(epsilon, minpts, NCHUNKS, clusterIDsArray, partDataPointsArray);

    return 0;
}