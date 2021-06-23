#include <algorithm>
#include <fstream>
#include <iostream>

#include "HybridDBSCAN.hpp"
#include "utils.hpp"

using std::cout;
using std::endl;
using std::ofstream;

HybridDBSCAN::HybridDBSCAN(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints,
    uint dataSize,
    uint blockSize,
    uint NCHUNKS) : DBSCAN(epsilon, minpts, dataPoints, dataSize),
                    blockSize(blockSize),
                    NCHUNKS(NCHUNKS),
                    gridSize(0) {
    printf("start HybridDBSCAN::HybridDBSCAN\n");
    if ((dataSize % NCHUNKS) != 0) {
        fprintf(stderr, "Error: dataSize mod NCHUNKS != 0\n");
    }
    clusterIDs.resize(dataSize);
    for (int i = 0; i < dataSize; i++) {
        clusterIDs[i] = HybridDBSCAN::UNVISITED;
    }
    neighborTables.resize(dataSize);
    valuePtrs.resize(NCHUNKS);

#pragma omp parallel for
    for (int i = 0; i < GPU_STREAMS; i++) {
        CHECK(cudaMallocHost((void **)&orderedIDKeys[i], sizeof(uint) * GPU_BUFFER_SIZE));
        CHECK(cudaMallocHost((void **)&orderedIDValues[i], sizeof(uint) * GPU_BUFFER_SIZE));
    }

    for (int i = 0; i < GPU_STREAMS; i++) {
        neighborsCnts[i] = 0;
    }
    printf("finish HybridDBSCAN::HybridDBSCAN\n");
}

HybridDBSCAN::~HybridDBSCAN() {
#pragma omp parallel for
    for (int i = 0; i < GPU_STREAMS; i++) {
        cudaFree(orderedIDKeys[i]);
        cudaFree(orderedIDValues[i]);
    }
    for (int i = 0; i < NCHUNKS; i++) {
        free(valuePtrs[i]);
    }
}

void HybridDBSCAN::run() {
    printf("start HybridDBSCAN::run\n");

    auto start = system_clock::now();
    calcCells(minVals, maxVals, nCells, totalCells);
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

    constructResultSetAndNeighborTable(
        NCHUNKS,
        nCells,
        gridSize,
        ordered2GridID,
        ordered2DataID,
        grid2CellID,
        index,
        blockSize,
        neighborsCnts,
        orderedIDKeys,
        orderedIDValues,
        valuePtrs,
        neighborTables);

    /*
    float gpu_elapsed_time_ms[NCHUNKS];
#pragma omp parallel for schedule(static, 1) num_threads(GPU_STREAMS)
    for (uint i = 0; i < NCHUNKS; i++) {
        printf("in run: i = %d\n", i);
        uint streamID = i % GPU_STREAMS;
        neighborsCnts[streamID] = 0;
        gpu_elapsed_time_ms[i] = constructGPUResultSet(
            i,
            NCHUNKS,
            nCells,
            gridSize,
            ordered2GridID,
            ordered2DataID,
            grid2CellID,
            index,
            blockSize,
            neighborsCnts[streamID],
            orderedIDKeys[streamID],
            orderedIDValues[streamID]);

        valuePtrs[i] = (uint *)malloc(sizeof(uint) * neighborsCnts[streamID]);

        constructNeighborTable(
            orderedIDKeys[streamID],
            orderedIDValues[streamID],
            neighborsCnts[streamID],
            valuePtrs[i],
            neighborTables);
    }
    */

    DBSCANwithNeighborTable(
        data2OrderedID,
        ordered2DataID,
        neighborTables,
        clusterIDs);

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Time elapsed on Hybrid-DBSCAN: "
         << double(duration.count()) * microseconds::period::num / milliseconds::period::den
         << "ms" << endl;

    printf("finish HybridDBSCAN::run\n");
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
        neighbors.push_back(ordered2DataID[neighborTables[orderedID].values[i]]);
    }
    sort(neighbors.begin(), neighbors.end());
    return neighbors;
}

void HybridDBSCAN::debug_printNeighborTable() {
    printf("in HybridDBSCAN::debug_printNeighborTable\n");

    for (uint i = 0; i < dataSize; i++) {
        uint orderedID = data2OrderedID[i];
        printf("\ndataID = %u, orderedID = %u, valueIdx_min = %d, valueIdx_max = %d\n", i, orderedID, neighborTables[orderedID].valueIdx_min, neighborTables[orderedID].valueIdx_max);
        for (int j = neighborTables[orderedID].valueIdx_min; j <= neighborTables[orderedID].valueIdx_max; j++) {
            printf("%d, ", ordered2DataID[neighborTables[orderedID].values[j]]);
        }
        printf("\n");
    }
    printf("\n");
    printf("finish HybridDBSCAN::debug_printNeighborTable\n");
}