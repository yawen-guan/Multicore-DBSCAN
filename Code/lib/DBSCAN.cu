#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <parallel/algorithm>

#include "DBSCAN.hpp"

using std::ceil;
using std::copy;
using std::cout;
using std::endl;
using std::iostream;
using std::lower_bound;
using std::max;
using std::min;
using std::ofstream;
using thrust::device_ptr;
using thrust::device_vector;
using thrust::raw_pointer_cast;
using thrust::sort_by_key;

DBSCAN::DBSCAN(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints,
    uint dataSize) : epsilon(epsilon),
                     epsilonPow(pow(epsilon, 2)),
                     minpts(minpts),
                     dataPoints(dataPoints),
                     dataSize(dataSize) {
}

DBSCAN::~DBSCAN() {
}

void DBSCAN::calcCells(
    array<float, 2> &minVals,
    array<float, 2> &maxVals,
    array<uint, 2> &nCells,
    uint64 &totalCells) {
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

void DBSCAN::constructIndex(
    const array<float, 2> &minVals,
    const array<float, 2> &maxVals,
    const array<uint, 2> &nCells,
    const uint64 &totalCells,
    uint &gridSize,
    vector<Grid> &index,
    vector<uint> &ordered2GridID,
    vector<uint> &ordered2DataID,
    vector<uint> &data2OrderedID,
    vector<uint> &grid2CellID) {
    printf("start DBSCAN::constructIndex\n");

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
    // std::sort(cellIDs_dup.begin(), cellIDs_dup.end());
    auto cellIDs_uqe = vector<uint64>();
    cellIDs_uqe.push_back(cellIDs_dup[0]);
    for (uint i = 1; i < cellIDs_dup.size(); i++) {
        if (cellIDs_dup[i] != cellIDs_dup[i - 1]) {
            cellIDs_uqe.push_back(cellIDs_dup[i]);
        }
    }

    grid2CellID.resize(cellIDs_uqe.size());
    for (uint i = 0; i < cellIDs_uqe.size(); i++) {
        grid2CellID[i] = cellIDs_uqe[i];
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

    printf("finish DBSCAN::constructIndex\n");
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
    const uint chunkID,
    const uint NCHUNKS,
    const uint chunkSize,
    const uint dataSize,
    const uint nCells1,
    const uint gridSize,
    const float epsilonPow,
    const float *d_dataPoints,
    const uint *d_ordered2DataID,
    const uint *d_ordered2GridID,
    const uint *d_grid2CellID,
    const Grid *d_index,
    uint *d_cnt,
    uint *d_orderedIDKey,
    uint *d_orderedIDValue) {
    uint globalID = blockIdx.x * blockDim.x + threadIdx.x;
    // if (globalID >= (dataSize / NCHUNKS)) return;
    if (globalID >= chunkSize) return;

    uint orderedID = globalID * NCHUNKS + chunkID;
    // uint dataID = d_ordered2DataID[orderedID];
    // printf("dataID = %u, orderedID = %u, globalID = %u, chunkID = %u\n", dataID, orderedID, globalID, chunkID);

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
                    if (orderedID == k) {
                        continue;
                    }
                    if (gpuInEpsilon(orderedID, k, epsilonPow, dataSize, d_dataPoints, d_ordered2DataID)) {
                        uint idx = atomicAdd(d_cnt, int(1));
                        d_orderedIDKey[idx] = orderedID;
                        d_orderedIDValue[idx] = k;
                    }
                }
            }
        }
    }
}

float DBSCAN::constructGPUResultSet(
    const uint &chunkID,
    const uint &NCHUNKS,
    const array<uint, 2> &nCells,
    const uint &gridSize,
    const vector<uint> &ordered2GridID,
    const vector<uint> &ordered2DataID,
    const vector<uint> &grid2CellID,
    const vector<Grid> &index,
    const uint &blockSize,
    uint &neighborsCnt,
    uint *&orderedIDKey,
    uint *&orderedIDValue) {
    float gpu_elapsed_time_ms;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    float *d_dataPoints;
    CHECK(cudaMalloc((void **)&d_dataPoints, sizeof(float) * 2 * dataSize));
    CHECK(cudaMemcpy(d_dataPoints, dataPoints[0].data(), sizeof(float) * dataSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dataPoints + dataSize, dataPoints[1].data(), sizeof(float) * dataSize, cudaMemcpyHostToDevice));

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

    uint *d_orderedIDKey;
    uint *d_orderedIDValue;
    CHECK(cudaMalloc((void **)&d_orderedIDKey, sizeof(uint) * GPU_BUFFER_SIZE));
    CHECK(cudaMalloc((void **)&d_orderedIDValue, sizeof(uint) * GPU_BUFFER_SIZE));

    uint *d_cnt;
    CHECK(cudaMalloc((void **)&d_cnt, sizeof(uint)));
    uint chunkSize = dataSize / NCHUNKS;

    dim3 dimGrid(ceil((double)dataSize / (double)NCHUNKS / (double)blockSize));
    dim3 dimBlock(blockSize);

    gpuCalcGlobal<<<dimGrid, dimBlock>>>(chunkID, NCHUNKS, chunkSize, dataSize, nCells[1], gridSize, epsilonPow, d_dataPoints, d_ordered2DataID, d_ordered2GridID, d_grid2CellID, d_index, d_cnt, d_orderedIDKey, d_orderedIDValue);
    cudaDeviceSynchronize();

    neighborsCnt = 0;
    cudaMemcpy(&neighborsCnt, d_cnt, sizeof(uint), cudaMemcpyDeviceToHost);

    // sort gpuResultSet
    device_ptr<uint> d_keyPtr(d_orderedIDKey);
    device_ptr<uint> d_valuePtr(d_orderedIDValue);

    try {
        sort_by_key(d_keyPtr, d_keyPtr + neighborsCnt, d_valuePtr);
    } catch (std::bad_alloc &e) {
        fprintf(stderr, "Error: Ran out of memory while sorting.\n");
        exit(-1);
    }

    cudaMemcpy(raw_pointer_cast(orderedIDKey), raw_pointer_cast(d_keyPtr), sizeof(uint) * neighborsCnt, cudaMemcpyDeviceToHost);
    cudaMemcpy(raw_pointer_cast(orderedIDValue), raw_pointer_cast(d_valuePtr), sizeof(uint) * neighborsCnt, cudaMemcpyDeviceToHost);

    cudaFree(d_dataPoints);
    cudaFree(d_ordered2DataID);
    cudaFree(d_ordered2GridID);
    cudaFree(d_grid2CellID);
    cudaFree(d_index);
    cudaFree(d_cnt);
    cudaFree(d_orderedIDKey);
    cudaFree(d_orderedIDValue);

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, end);

    return gpu_elapsed_time_ms;
}

void DBSCAN::constructNeighborTable(
    const uint *orderedIDKey,
    const uint *orderedIDValue,
    const uint &neighborsCnt,
    uint *&valuePtr,
    vector<NeighborTable> &neighborTables) {
    copy(orderedIDValue, orderedIDValue + neighborsCnt, valuePtr);

    auto keyData_uqe = vector<KeyData>();
    keyData_uqe.push_back(KeyData(orderedIDKey[0], 0));
    for (uint i = 1; i < neighborsCnt; i++) {
        if (orderedIDKey[i] != orderedIDKey[i - 1]) {
            keyData_uqe.push_back(KeyData(orderedIDKey[i], i));
        }
    }

    // neighborTables.resize(dataSize);
    uint key = 0;
    for (uint i = 0; i < keyData_uqe.size(); i++) {
        key = keyData_uqe[i].key;
        neighborTables[key].values = valuePtr;
        neighborTables[key].valueIdx_min = keyData_uqe[i].pos;
        if (i == keyData_uqe.size() - 1) {
            neighborTables[key].valueIdx_max = neighborsCnt - 1;
        } else {
            neighborTables[key].valueIdx_max = keyData_uqe[i + 1].pos - 1;
        }
    }
}

void DBSCAN::constructResultSetAndNeighborTable(
    const uint &NCHUNKS,
    const array<uint, 2> &nCells,
    const uint &gridSize,
    const vector<uint> &ordered2GridID,
    const vector<uint> &ordered2DataID,
    const vector<uint> &grid2CellID,
    const vector<Grid> &index,
    const uint &blockSize,
    uint neighborsCnts[GPU_STREAMS],
    uint *orderedIDKeys[GPU_STREAMS],
    uint *orderedIDValues[GPU_STREAMS],
    vector<uint *> &valuePtrs,
    vector<NeighborTable> &neighborTables) {
    printf("start DBSCAN::constructResultSetAndNeighborTable\n");

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

    uint *d_orderedIDKeys[GPU_STREAMS];
    uint *d_orderedIDValues[GPU_STREAMS];
    for (uint i = 0; i < GPU_STREAMS; i++) {
        CHECK(cudaMalloc((void **)&d_orderedIDKeys[i], sizeof(uint) * GPU_BUFFER_SIZE));
        CHECK(cudaMalloc((void **)&d_orderedIDValues[i], sizeof(uint) * GPU_BUFFER_SIZE));
    }

    uint *d_cnt;
    CHECK(cudaMalloc((void **)&d_cnt, sizeof(uint) * GPU_STREAMS));

    uint chunkSize = dataSize / NCHUNKS;

    cudaStream_t stream[GPU_STREAMS];
    for (uint i = 0; i < GPU_STREAMS; i++) {
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }

// #pragma omp parallel for schedule(static, 1) num_threads(GPU_STREAMS)
#pragma omp parallel for schedule(static, 1) num_threads(GPU_STREAMS)
    for (uint i = 0; i < NCHUNKS; i++) {
        uint chunkID = i;
        int streamID = omp_get_thread_num();
        printf("chunkID = %u, streamID = %d\n", chunkID, streamID);

        neighborsCnts[streamID] = 0;
        CHECK(cudaMemcpyAsync(&d_cnt[streamID], &neighborsCnts[streamID], sizeof(uint), cudaMemcpyHostToDevice, stream[streamID]));

        dim3 dimGrid(ceil((double)chunkSize / (double)blockSize));
        dim3 dimBlock(blockSize);
        gpuCalcGlobal<<<dimGrid, dimBlock, 0, stream[streamID]>>>(chunkID, NCHUNKS, chunkSize, dataSize, nCells[1], gridSize, epsilonPow, d_dataPoints, d_ordered2DataID, d_ordered2GridID, d_grid2CellID, d_index, &d_cnt[chunkID], d_orderedIDKeys[streamID], d_orderedIDValues[streamID]);

        CHECK(cudaMemcpyAsync(&neighborsCnts[streamID], &d_cnt[streamID], sizeof(uint), cudaMemcpyDeviceToHost, stream[streamID]));

        // sort gpuResultSet
        device_ptr<uint> d_keyPtr(d_orderedIDKeys[streamID]);
        device_ptr<uint> d_valuePtr(d_orderedIDValues[streamID]);

        try {
            sort_by_key(thrust::cuda::par.on(stream[streamID]), d_keyPtr, d_keyPtr + neighborsCnts[streamID], d_valuePtr);
        } catch (std::bad_alloc &e) {
            fprintf(stderr, "Error: Ran out of memory while sorting.\n");
            exit(-1);
        }

        cudaMemcpyAsync(raw_pointer_cast(orderedIDKeys[streamID]), raw_pointer_cast(d_keyPtr), sizeof(uint) * neighborsCnts[streamID], cudaMemcpyDeviceToHost, stream[streamID]);
        cudaMemcpyAsync(raw_pointer_cast(orderedIDValues[streamID]), raw_pointer_cast(d_valuePtr), sizeof(uint) * neighborsCnts[streamID], cudaMemcpyDeviceToHost, stream[streamID]);

        cudaStreamSynchronize(stream[streamID]);

        valuePtrs[chunkID] = new uint[neighborsCnts[streamID]];
        constructNeighborTable(orderedIDKeys[streamID], orderedIDValues[streamID], neighborsCnts[streamID], valuePtrs[chunkID], neighborTables);
    }

    cudaFree(d_dataPoints);
    cudaFree(d_ordered2DataID);
    cudaFree(d_ordered2GridID);
    cudaFree(d_grid2CellID);
    cudaFree(d_index);
    cudaFree(d_cnt);
    for (uint i = 0; i < GPU_STREAMS; i++) {
        cudaFree(d_orderedIDKeys[i]);
        cudaFree(d_orderedIDValues[i]);
    }

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, end);

    for (int i = 0; i < GPU_STREAMS; i++) {
        CHECK(cudaStreamDestroy(stream[i]));
    }

    printf("finish DBSCAN::constructResultSetAndNeighborTable\n");
    // return gpu_elapsed_time_ms;
}

void DBSCAN::DBSCANwithNeighborTable(
    const vector<uint> &data2OrderedID,
    const vector<uint> &ordered2DataID,
    const vector<NeighborTable> &neighborTables,
    vector<int> &clusterIDs) {
    auto neighbors = vector<uint>();
    int clusterID = 0, orderedID = -1, size = 0, newSize = 0;

    for (uint i = 0; i < dataSize; i++) {
        orderedID = data2OrderedID[i];

        if (clusterIDs[i] != DBSCAN::UNVISITED) {
            continue;
        }
        neighbors.clear();

        if (neighborTables[orderedID].valueIdx_min == -1 || neighborTables[orderedID].valueIdx_max == -1) {
            size = 0;
        } else {
            size = neighborTables[orderedID].valueIdx_max - neighborTables[orderedID].valueIdx_min + 1;
        }

        if ((size + 1) < minpts) {
            clusterIDs[i] = DBSCAN::NOISE;
        } else {
            clusterIDs[i] = ++clusterID;
            neighbors.resize(size);
            for (uint j = neighborTables[orderedID].valueIdx_min; j <= neighborTables[orderedID].valueIdx_max; j++) {
                neighbors[j - neighborTables[orderedID].valueIdx_min] = neighborTables[orderedID].values[j];  // orderedIDValue[j];
            }

            while (neighbors.size() != 0) {
                uint pOrderedID = neighbors.back();
                uint p = ordered2DataID[pOrderedID];

                if (clusterIDs[p] == DBSCAN::UNVISITED) {
                    if (neighborTables[pOrderedID].valueIdx_min == -1 || neighborTables[pOrderedID].valueIdx_max == -1) {
                        newSize = 0;
                    } else {
                        newSize = neighborTables[pOrderedID].valueIdx_max - neighborTables[pOrderedID].valueIdx_min + 1;
                    }

                    if ((newSize + 1) >= minpts) {
                        neighbors.resize(size + newSize);
                        for (uint j = neighborTables[pOrderedID].valueIdx_min; j <= neighborTables[pOrderedID].valueIdx_max; j++) {
                            neighbors[size + j - neighborTables[pOrderedID].valueIdx_min] = neighborTables[pOrderedID].values[j];  // orderedIDValue[j];
                        }
                    }
                }

                if (clusterIDs[p] == DBSCAN::UNVISITED || clusterIDs[p] == DBSCAN::NOISE) {
                    clusterIDs[p] = clusterID;
                }

                neighbors.pop_back();
            }
        }
    }
}