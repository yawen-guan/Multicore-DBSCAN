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

#include "Shadower.cuh"

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

Shadower::Shadower(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints,
    uint dataSize,
    uint blockSize, 
    vector<vector<int>> clusterIDsArray, 
    vector<uint> pointIDs_shadow, 
    vector<PointChunkLookup> pointChunkMapping) : epsilon(epsilon),
                      epsilonPow(pow(epsilon, 2)),
                      minpts(minpts),
                      dataPoints(dataPoints),
                      dataSize(dataSize),
                      blockSize(blockSize),
                      gridSize(0),
                      orderedIDKey(nullptr),
                      orderedIDValue(nullptr),
                      neighborsCnt(0),
                      clusterIDsArray(clusterIDsArray), 
                      pointIDs_shadow(pointIDs_shadow), 
                      pointChunkMapping(pointChunkMapping) {
    clusterIDs.resize(dataSize);
    for (int i = 0; i < dataSize; i++) {
        clusterIDs[i] = Shadower::UNVISITED;
    }
}

void Shadower::calcCells() {
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
    totalCells = 1;
    for (uint dim = 0; dim < 2; dim++) {
        nCells[dim] = ceil((maxVals[dim] - minVals[dim]) / epsilon);
        totalCells *= nCells[dim];
    }
}

void Shadower::constructIndex() {
    calcCells();
    auto cellIDs_dup = vector<uint64>();
    for (uint i = 0; i < dataSize; i++) {
        uint64 l0 = (dataPoints[0][i] - minVals[0]) / epsilon;
        uint64 l1 = (dataPoints[1][i] - minVals[1]) / epsilon;
        uint64 cellID = l0 * nCells[1] + l1;
        cellIDs_dup.push_back(cellID);
    }
    std::sort(cellIDs_dup.begin(), cellIDs_dup.end());
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
    gridSize = cellIDs_uqe.size();
    auto gridDataIDs = vector<vector<uint64>>(gridSize); 
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

__device__ int findGridByCellID_s(const uint cellID, const uint gridSize, const uint *d_grid2CellID) {
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

__device__ bool gpuInEpsilon_s(
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

__global__ void gpuCalcGlobal_s(
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
    uint orderedID = blockIdx.x * blockDim.x + threadIdx.x;
    if (orderedID >= dataSize) return;

    uint gridID = d_ordered2GridID[orderedID];
    uint cellID = d_grid2CellID[gridID];
    uint newCellID = 0;

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) {
                gridID = d_ordered2GridID[orderedID];
            } else {
                newCellID = cellID + (i * nCells1 + j);
                gridID = findGridByCellID_s(newCellID, gridSize, d_grid2CellID);
            }

            if (gridID < gridSize) {
                for (uint k = d_index[gridID].orderedID_min; k <= d_index[gridID].orderedID_max; k++) {
                    if (orderedID == k) {
                        continue;
                    }
                    if (gpuInEpsilon_s(orderedID, k, epsilonPow, dataSize, d_dataPoints, d_ordered2DataID)) {
                        uint idx = atomicAdd(d_cnt, int(1));
                        d_orderedIDKey[idx] = orderedID;
                        d_orderedIDValue[idx] = k;
                    }
                }
            }
        }
    }
}

float Shadower::constructGPUResultSet() {
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

    uint *d_orderedIDKey;    //key
    uint *d_orderedIDValue;  //value
    cudaMalloc((void **)&d_orderedIDKey, sizeof(uint) * GPU_BUFFER_SIZE);
    cudaMalloc((void **)&d_orderedIDValue, sizeof(uint) * GPU_BUFFER_SIZE);
    cudaMallocHost((void **)&orderedIDKey, sizeof(uint) * GPU_BUFFER_SIZE);
    cudaMallocHost((void **)&orderedIDValue, sizeof(uint) * GPU_BUFFER_SIZE);

    uint *d_cnt;
    cudaMalloc((void **)&d_cnt, sizeof(uint));

    dim3 dimGrid(ceil((double)dataSize / (double)blockSize));
    dim3 dimBlock(blockSize);

    gpuCalcGlobal_s<<<dimGrid, dimBlock>>>(dataSize, nCells[1], gridSize, epsilonPow, d_dataPoints, d_ordered2DataID, d_ordered2GridID, d_grid2CellID, d_index, d_cnt, d_orderedIDKey, d_orderedIDValue);
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

void Shadower::constructNeighborTable() {
    auto keyData_uqe = vector<KeyData>();
    keyData_uqe.push_back(KeyData(orderedIDKey[0], 0));
    for (uint i = 1; i < neighborsCnt; i++) {
        if (orderedIDKey[i] != orderedIDKey[i - 1]) {
            keyData_uqe.push_back(KeyData(orderedIDKey[i], i));
        }
    }

    neighborTables.resize(dataSize);
    uint key = 0;
    for (uint i = 0; i < keyData_uqe.size(); i++) {
        key = keyData_uqe[i].key;
        neighborTables[key].valueIdx_min = keyData_uqe[i].pos;
        if (i == keyData_uqe.size() - 1) {
            neighborTables[key].valueIdx_max = neighborsCnt - 1;
        } else {
            neighborTables[key].valueIdx_max = keyData_uqe[i + 1].pos - 1;
        }
    }
}

void Shadower::modifiedDBSCAN() {
    auto neighbors = vector<uint>();
    int clusterID = 0, orderedID = -1;

    for (uint i = 0; i < dataSize; i++) {
        orderedID = data2OrderedID[i];

        if (clusterIDs[i] != Shadower::UNVISITED) {
            continue;
        }
        neighbors.clear();

        uint size = neighborTables[orderedID].valueIdx_max - neighborTables[orderedID].valueIdx_min + 1;

        if ((size + 1) < minpts) {
            clusterIDs[i] = Shadower::NOISE;
            neighbors.resize(size);
            for (uint j = neighborTables[orderedID].valueIdx_min; j <= neighborTables[orderedID].valueIdx_max; j++) {
                neighbors[j - neighborTables[orderedID].valueIdx_min] = orderedIDValue[j];
            }

            while (neighbors.size() != 0) {
                uint pOrderedID = neighbors.back();
                uint p = ordered2DataID[pOrderedID];
                
                int pID = clusterIDsArray[pointChunkMapping[pointIDs_shadow[p]].chunkID][pointChunkMapping[pointIDs_shadow[p]].idxInChunk];
                int iID = clusterIDsArray[pointChunkMapping[pointIDs_shadow[i]].chunkID][pointChunkMapping[pointIDs_shadow[i]].idxInChunk];
                if(merge.find(pID)==merge.end()){
                    int t = iID;
                    bool f = true;
                    while(merge.find(t)!= merge.end()){
                        if(t == pID){
                            f = false;
                            break;
                        }
                        t = merge[t];
                    }
                    if(f&&t != pID)merge.emplace(pID,t);
                }

                neighbors.pop_back();
            }
        } else {
            clusterIDs[i] = ++clusterID;
            neighbors.resize(size);
            for (uint j = neighborTables[orderedID].valueIdx_min; j <= neighborTables[orderedID].valueIdx_max; j++) {
                neighbors[j - neighborTables[orderedID].valueIdx_min] = orderedIDValue[j];
            }

            while (neighbors.size() != 0) {
                uint pOrderedID = neighbors.back();
                uint p = ordered2DataID[pOrderedID];

                if (clusterIDs[p] == Shadower::UNVISITED) {
                    uint newSize = neighborTables[pOrderedID].valueIdx_max - neighborTables[pOrderedID].valueIdx_min + 1;
                    if ((newSize + 1) >= minpts) {
                        neighbors.resize(size + newSize);
                        for (uint j = neighborTables[pOrderedID].valueIdx_min; j <= neighborTables[pOrderedID].valueIdx_max; j++) {
                            neighbors[size + j - neighborTables[pOrderedID].valueIdx_min] = orderedIDValue[j];
                        }
                    }
                }

                int pID = clusterIDsArray[pointChunkMapping[pointIDs_shadow[p]].chunkID][pointChunkMapping[pointIDs_shadow[p]].idxInChunk];
                int iID = clusterIDsArray[pointChunkMapping[pointIDs_shadow[i]].chunkID][pointChunkMapping[pointIDs_shadow[i]].idxInChunk];
                if(merge.find(pID)==merge.end()){
                    int t = iID;
                    bool f = true;
                    while(merge.find(t)!= merge.end()){
                        if(t == pID){
                            f = false;
                            break;
                        }
                        t = merge[t];
                    }
                    if(f&&t != pID)merge.emplace(pID,t);
                }
                

                if (clusterIDs[p] == Shadower::UNVISITED || clusterIDs[p] == Shadower::NOISE) {
                    clusterIDs[p] = clusterID;
                }

                neighbors.pop_back();
            }
        }
    }
}

void Shadower::run() {
    constructIndex();
    float gpu_elapsed_time_ms = constructGPUResultSet();
    constructNeighborTable();
    modifiedDBSCAN();
}