#include "DBSCAN.hpp"

#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::iostream;
using std::ofstream;

DBSCAN::DBSCAN(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints,
    uint dataSize) : epsilon(epsilon),
                     epsilonPow(pow(epsilon, 2)),
                     minpts(minpts),
                     dataPoints(dataPoints),
                     dataSize(dataSize) {
    clusterIDs.resize(dataSize);
    for (int i = 0; i < dataSize; i++) {
        clusterIDs[i] = DBSCAN::UNVISITED;
    }
}

void DBSCAN::run() {
    float start = omp_get_wtime();

    int clusterID = 0;
    for (uint i = 0; i < dataSize; i++) {
        if (clusterIDs[i] == DBSCAN::UNVISITED) {
            auto neighbors = getNeighbors(i);
            if ((neighbors.size() + 1) < minpts) {
                clusterIDs[i] = DBSCAN::NOISE;
            } else {
                clusterIDs[i] = ++clusterID;
                for (uint j = 0; j < neighbors.size(); j++) {
                    uint p = neighbors[j];
                    if (clusterIDs[p] == DBSCAN::UNVISITED) {
                        auto pNeighbors = getNeighbors(p);
                        if ((pNeighbors.size() + 1) >= minpts) {
                            neighbors.insert(neighbors.end(), pNeighbors.begin(), pNeighbors.end());
                        }
                    }
                    if (clusterIDs[p] == DBSCAN::UNVISITED || clusterIDs[p] == DBSCAN::NOISE) {
                        clusterIDs[p] = clusterID;
                    }
                }
            }
        }
    }
    float end = omp_get_wtime();
    float elapsed_time_ms = (end - start) * 1000;
    printf("Time elapsed on original DBSCAN: %f ms\n", elapsed_time_ms);
}

float DBSCAN::dist(const uint& id0, const uint& id1) {
    return pow(dataPoints[0][id0] - dataPoints[0][id1], 2) + pow(dataPoints[1][id0] - dataPoints[1][id1], 2);
}

vector<uint> DBSCAN::getNeighbors(const uint& id) {
    auto neighbors = vector<uint>();
    neighbors.clear();
    for (uint i = 0; i < dataSize; i++) {
        if (i == id) {
            continue;
        } else if (DBSCAN::dist(id, i) <= epsilonPow) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

void DBSCAN::print(const string& outFile) {
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

void DBSCAN::debug_printNeighborTable() {
    for (uint i = 0; i < dataSize; i++) {
        printf("point %u, neighbors:", i);
        auto neighbors = getNeighbors(i);
        for (uint j = 0; j < neighbors.size(); j++) {
            printf("%u, ", neighbors[j]);
        }
    }
    printf("\n");
}