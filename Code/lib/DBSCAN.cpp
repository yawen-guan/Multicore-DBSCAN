#include "DBSCAN.hpp"

#include <cmath>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::iostream;
using std::ofstream;
using std::pow;

DBSCAN::DBSCAN(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints,
    uint dataSize) : epsilon(epsilon), epsilonPow(pow(epsilon, 2)), minpts(minpts), dataPoints(dataPoints), dataSize(dataSize) {
    this->clusterIDs.resize(dataSize);
    for (int i = 0; i < dataSize; i++) {
        clusterIDs[i] = DBSCAN::UNVISITED;
    }
}

void DBSCAN::run() {
    uint clusterID = 0;
    for (uint i = 0; i < dataSize; i++) {
        if (clusterIDs[i] == DBSCAN::UNVISITED) {
            auto neighbors = DBSCAN::getNeighbors(i);
            if (neighbors.size() < this->minpts) {
                clusterIDs[i] = DBSCAN::NOISE;
            } else {
                clusterIDs[i] = ++clusterID;
                for (uint j = 0; j < neighbors.size(); j++) {
                    uint p = neighbors[j];
                    if (clusterIDs[p] == DBSCAN::UNVISITED) {
                        auto pNeighbors = DBSCAN::getNeighbors(p);
                        if (pNeighbors.size() >= this->minpts) {
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
}

float DBSCAN::dist(const uint& id0, const uint& id1) {
    return pow(this->dataPoints[0][id0] - this->dataPoints[0][id1], 2) + pow(this->dataPoints[1][id0] - this->dataPoints[1][id1], 2);
}

vector<uint> DBSCAN::getNeighbors(const uint& id) {
    auto neighbors = vector<uint>();
    for (uint i = 0; i < this->dataSize; i++) {
        if (DBSCAN::dist(id, i) <= this->epsilonPow) {
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
        for (uint i = 0; i < this->dataSize; i++) {
            sprintf(buffer, "%5.15lf,%5.15lf,%d\n", dataPoints[0][i], dataPoints[1][i], clusterIDs[i]);
            out << buffer;
        }
        out.close();
    } else {
        cout << "Unable to open file " << outFile << endl;
    }
}