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
#include "BPSDBSCAN.hpp"
#include "global.hpp"
#include "utils.hpp"

using std::cin;
using std::cout;
using std::endl;

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
    cout << "datafile = " << datafile << ", epsilon = " << epsilon << ", minpts = " << minpts << ", NCHUNK = " << NCHUNKS << ", blockSize = " << blockSize << endl;

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

    /***** BPS-DBSCAN *****/

    auto BPS_dbscan = BPSDBSCAN(epsilon, minpts, dataPoints, dataSize, blockSize, NCHUNKS);
    BPS_dbscan.run();
    printf("\n----------- BPS DBSCAN -----------\n");
    BPS_dbscan.print("../data/output/BPS-DBSCAN-data-2500-out.csv");

    check(dataSize, dbscan.clusterIDs, BPS_dbscan.finalClusterIDs);

    return 0;
}