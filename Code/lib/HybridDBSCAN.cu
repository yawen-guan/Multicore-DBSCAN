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
    uint blockSize) : epsilon(epsilon),
                      epsilonPow(pow(epsilon, 2)),
                      minpts(minpts),
                      dataPoints(dataPoints),
                      dataSize(dataSize),
                      blockSize(blockSize),
                      gridSize(0) {
    clusterIDs.resize(dataSize);
    for (int i = 0; i < dataSize; i++) {
        clusterIDs[i] = HybridDBSCAN::UNVISITED;
    }
}

void HybridDBSCAN::calcCells() {
    // minVals, maxVals
    for (uint dim = 0; dim < 2; dim++) {
        minVals[dim] = dataPoints[dim][0];
        maxVals[dim] = dataPoints[dim][0];
    }
    for (uint dim = 0; dim < 2; dim++) {
        for (uint i = 0; i < dataSize; i++) {
            minVals[dim] = min(minVals[dim], dataPoints[dim][i]);
            maxVals[dim] = max(maxVals[dim], dataPoints[dim][i]);
        }
    }
    for (uint dim = 0; dim < 2; dim++) {
        minVals[dim] -= epsilon;
        maxVals[dim] += epsilon;
    }
    // nCells
    totalCells = 1;
    for (uint dim = 0; dim < 2; dim++) {
        nCells[dim] = ceil((maxVals[dim] - minVals[dim]) / epsilon);
        totalCells *= nCells[dim];
    }
}

void HybridDBSCAN::constructIndex() {
    calcCells();

    // calculate cellIDs of grid cells
    auto cellIDs_dup = vector<uint64>();
    for (uint i = 0; i < dataSize; i++) {
        uint64 l0 = (dataPoints[0][i] - minVals[0]) / epsilon;
        uint64 l1 = (dataPoints[1][i] - minVals[1]) / epsilon;
        uint64 cellID = l0 * nCells[1] + l1;
        cellIDs_dup.push_back(cellID);
    }
    omp_set_num_threads(NUM_THREADS);
    __gnu_parallel::sort(cellIDs_dup.begin(), cellIDs_dup.end());
    auto cellIDs_uqe = vector<uint64>();
    cellIDs_uqe.push_back(cellIDs_dup[0]);
    for (uint i = 1; i < cellIDs_dup.size(); i++) {
        if (cellIDs_dup[i] != cellIDs_dup[i - 1]) {
            cellIDs_uqe.push_back(cellIDs_dup[i]);
        }
    }

    grid2CellID.resize(cellIDs_uqe.size());
    cell2GridID.clear();
    for (uint i = 0; i < cellIDs_uqe.size(); i++) {
        grid2CellID[i] = cellIDs_uqe[i];
        cell2GridID[cellIDs_uqe[i]] = i;
    }

    // get dataIDs per non empty cell(aka. grid)
    gridSize = cellIDs_uqe.size();
    auto gridDataIDs = vector<vector<uint64>>(gridSize);  // dataID per cell
    for (uint i = 0; i < dataSize; i++) {
        uint64 l0 = (dataPoints[0][i] - minVals[0]) / epsilon;
        uint64 l1 = (dataPoints[1][i] - minVals[1]) / epsilon;
        uint64 cellID = l0 * nCells[1] + l1;
        if (cellID > totalCells) {
            fprintf(stderr, "ERROR: LinearID = %lu > totalCells = %lu\n", cellID, totalCells);
        }

        vector<uint64>::iterator lower = lower_bound(cellIDs_uqe.begin(), cellIDs_uqe.end(), cellID);
        uint64 gridID = lower - cellIDs_uqe.begin();
        gridDataIDs[gridID].push_back(i);
    }

    // construct index
    index.resize(gridSize);
    ordered2DataID.resize(dataSize);
    data2OrderedID.resize(dataSize);
    ordered2GridID.resize(dataSize);
    uint orderedID = 0;
    for (uint i = 0; i < gridSize; i++) {
        index[i].orderedID_min = orderedID;
        for (uint j = 0; j < gridDataIDs[i].size(); j++) {
            ordered2DataID[orderedID] = gridDataIDs[i][j];
            data2OrderedID[gridDataIDs[i][j]] = orderedID;
            ordered2GridID[orderedID] = i;
            orderedID++;
        }
        index[i].orderedID_max = orderedID - 1;
    }
}

__device__ int findGridByCellID(const uint cellID, const uint gridSize, const uint *d_grid2CellID) {
    int l = 0, r = gridSize - 1;
    int mid = (l + r) / 2;
    while (l <= r) {
        if (d_grid2CellID[mid] < cellID) {
            l = mid + 1;
        } else if (d_grid2CellID[mid] > cellID) {
            r = mid - 1;
        } else if (d_grid2CellID[mid] == cellID) {
            return mid;
        }
        mid = (l + r) / 2;
    }
    return gridSize + 1;
}

__device__ bool gpuInEpsilon(
    const uint orderedID0,
    const uint orderedID1,
    const float epsilonPow,
    const uint dataSize,
    const float *d_dataPoints,
    const uint *d_ordered2DataID) {
    uint id0 = d_ordered2DataID[orderedID0];
    uint id1 = d_ordered2DataID[orderedID1];
    return (pow(d_dataPoints[id0] - d_dataPoints[id1], 2) + pow(d_dataPoints[dataSize + id0] - d_dataPoints[dataSize + id1], 2)) <= epsilonPow;
}

__global__ void gpuCalcGlobal(
    const uint dataSize,
    const uint nCells1,
    const uint gridSize,
    const float epsilonPow,
    const float *d_dataPoints,
    const uint *d_ordered2DataID,
    const uint *d_ordered2GridID,
    const uint *d_grid2CellID,
    const Grid *d_index,
    uint *d_neighborTable) {
    uint orderedID = blockIdx.x * blockDim.x + threadIdx.x;
    if (orderedID >= dataSize) return;
    // uint dataID = d_ordered2DataID[orderedID];

    uint cnt = 0;
    uint gridID = d_ordered2GridID[orderedID];
    uint cellID = d_grid2CellID[gridID];
    uint newCellID = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                gridID = d_ordered2GridID[orderedID];
            } else {
                newCellID = cellID + (i * nCells1 + j);
                // or a cell2GridID map
                gridID = findGridByCellID(newCellID, gridSize, d_grid2CellID);
            }

            if (gridID < gridSize) {
                for (uint k = d_index[gridID].orderedID_min; k <= d_index[gridID].orderedID_max; k++) {
                    if (gpuInEpsilon(orderedID, k, epsilonPow, dataSize, d_dataPoints, d_ordered2DataID)) {
                        cnt++;
                        d_neighborTable[orderedID * dataSize + cnt] = k;
                    }
                }
            }
        }
    }
    d_neighborTable[orderedID * dataSize + 0] = cnt;
}

float HybridDBSCAN::constructGPUResultSet() {
    float gpu_elapsed_time_ms;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    float *d_dataPoints;
    cudaMalloc((void **)&d_dataPoints, sizeof(float) * 2 * dataSize);
    cudaMemcpy(d_dataPoints, dataPoints[0].data(), sizeof(float) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataPoints + dataSize, dataPoints[1].data(), sizeof(float) * dataSize, cudaMemcpyHostToDevice);

    uint *d_ordered2DataID, *d_ordered2GridID, *d_grid2CellID;
    cudaMalloc((void **)&d_ordered2DataID, sizeof(uint) * dataSize);
    cudaMalloc((void **)&d_ordered2GridID, sizeof(uint) * dataSize);
    cudaMalloc((void **)&d_grid2CellID, sizeof(uint) * dataSize);
    cudaMemcpy(d_ordered2DataID, ordered2DataID.data(), sizeof(uint) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ordered2GridID, ordered2GridID.data(), sizeof(uint) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid2CellID, grid2CellID.data(), sizeof(uint) * gridSize, cudaMemcpyHostToDevice);

    Grid *d_index;
    cudaMalloc((void **)&d_index, sizeof(Grid) * dataSize);
    cudaMemcpy(d_index, index.data(), sizeof(Grid) * dataSize, cudaMemcpyHostToDevice);

    uint *d_neighborTable;
    cudaMalloc((void **)&d_neighborTable, sizeof(uint) * dataSize * dataSize);

    dim3 dimGrid(ceil((double)dataSize / (double)blockSize));
    dim3 dimBlock(blockSize);

    gpuCalcGlobal<<<dimGrid, dimBlock>>>(dataSize, nCells[1], gridSize, epsilonPow, d_dataPoints, d_ordered2DataID, d_ordered2GridID, d_grid2CellID, d_index, d_neighborTable);
    cudaDeviceSynchronize();

    uint *h_neighborTable = (uint *)malloc(sizeof(uint) * dataSize * dataSize);
    cudaMemcpy(h_neighborTable, d_neighborTable, sizeof(uint) * dataSize * dataSize, cudaMemcpyDeviceToHost);

    cudaFree(d_dataPoints);
    cudaFree(d_ordered2DataID);
    cudaFree(d_ordered2GridID);
    cudaFree(d_grid2CellID);
    cudaFree(d_index);
    cudaFree(d_neighborTable);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, end);

    neighborTable.resize(dataSize);
    for (uint i = 0; i < dataSize; i++) {
        uint *start = h_neighborTable + i * dataSize;
        uint size = start[0];
        neighborTable[i] = vector<uint>(start + 1, start + size + 1);
    }
    free(h_neighborTable);

    return gpu_elapsed_time_ms;
}

void HybridDBSCAN::modifiedDBSCAN() {
    int clusterID = 0, orderedID = -1;
    for (uint i = 0; i < dataSize; i++) {
        orderedID = data2OrderedID[i];

        if (clusterIDs[i] == HybridDBSCAN::UNVISITED) {
            auto neighbors = neighborTable[orderedID];
            if (neighbors.size() < minpts) {
                clusterIDs[i] = HybridDBSCAN::NOISE;
            } else {
                clusterIDs[i] = ++clusterID;
                for (uint j = 0; j < neighbors.size(); j++) {
                    // printf("j = %u, size = %d\n", j, neighbors.size());
                    uint pOrderedID = neighbors[j];
                    uint p = ordered2DataID[pOrderedID];

                    if (clusterIDs[p] == HybridDBSCAN::UNVISITED) {
                        auto pNeighbors = neighborTable[pOrderedID];
                        if (pNeighbors.size() >= minpts) {
                            neighbors.insert(neighbors.end(), pNeighbors.begin(), pNeighbors.end());
                        }
                    }
                    if (clusterIDs[p] == HybridDBSCAN::UNVISITED || clusterIDs[p] == HybridDBSCAN::NOISE) {
                        clusterIDs[p] = clusterID;
                    }
                }
            }
        }
    }
}

void HybridDBSCAN::run() {
    float start = omp_get_wtime();
    constructIndex();
    float gpu_elapsed_time_ms = constructGPUResultSet();
    modifiedDBSCAN();
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

void HybridDBSCAN::debug_printNeighborTable() {
    for (int i = 0; i < dataSize; i++) {
        uint orderedID = data2OrderedID[i];
        printf("\ndataID = %d, neighbors's dataID: ", i);
        for (int j = 0; j < neighborTable[orderedID].size(); j++) {
            printf("%d, ", ordered2DataID[neighborTable[orderedID][j]]);
        }
    }
    printf("\n");
}