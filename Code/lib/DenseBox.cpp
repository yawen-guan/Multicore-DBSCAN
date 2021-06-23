#include "DenseBox.hpp"

void DenseBox::calcCells() {
    // _minVals, _maxVals
    for (uint dim = 0; dim < 2; dim++) {
        _minVals[dim] = _dataPoints[dim][0];
        _maxVals[dim] = _dataPoints[dim][0];
    }
    for (uint dim = 0; dim < 2; dim++) {
        for (uint i = 0; i < _dataSize; i++) {
            _minVals[dim] = min(_minVals[dim], _dataPoints[dim][i]);
            _maxVals[dim] = max(_maxVals[dim], _dataPoints[dim][i]);
        }
    }
    for (uint dim = 0; dim < 2; dim++) {
        _minVals[dim] -= _epsilon;
        _maxVals[dim] += _epsilon;
    }
    // nCells
    _totalCells = 1;
    for (uint dim = 0; dim < 2; dim++) {
        _nCells[dim] = ceil((_maxVals[dim] - _minVals[dim]) / _epsilon);
        _totalCells *= _nCells[dim];
    }
}

DenseBox::DenseBox(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints) : _epsilon(epsilon / 2.828),
                                 _oriEpsilon(epsilon),
                                 _minpts(minpts),
                                 _dataPoints(dataPoints),
                                 _dataSize(dataPoints[0].size()),
                                 _curClusterNums(0) {
    calcCells();
    fprintf(stdout,
            "%lu cells, dim0 minVal = %f, maxVal= %f, dim1 minVal = %f, maxVal = %f\n",
            _totalCells, _minVals[0], _maxVals[0], _minVals[1], _maxVals[1]);
    _cellInners.resize(_totalCells);
    _DBCluster.resize(_dataSize);
    _densePointsID2dataPointsID.resize(_dataSize);
}

void DenseBox::clusterAndGetDensePoints() {
    for (uint i = 0; i < _dataSize; i++) {
        uint64 l0 = (_dataPoints[0][i] - _minVals[0]) / _epsilon;
        uint64 l1 = (_dataPoints[1][i] - _minVals[1]) / _epsilon;
        uint64 cellID = l0 * _nCells[1] + l1;
        _cellInners[cellID].push_back(i);
        _point2ID[Point(_dataPoints[0][i], _dataPoints[1][i])] = i;
    }
    auto denseID = 0;
    for (auto i = 0; i < _totalCells; ++i) {
        auto _size = _cellInners[i].size();
        if (!_size)
            continue;
        else if (_size > _minpts) {
            for (auto pt : _cellInners[i]) _DBCluster[pt] = i;
            _nonEmptyCellIDs.push_back(i);
            ++_curClusterNums;
        } else {
            for (auto pt : _cellInners[i]) {
                _densePoints[0].push_back(_dataPoints[0][pt]);
                _densePoints[1].push_back(_dataPoints[1][pt]);
                _densePointsID2dataPointsID[denseID++] = pt;
            }
            _nonEmptyCellIDs.push_back(i);
        }
    }
    fprintf(stdout, "densePointsSize = %lu, emptyCells = %lu\n", _densePoints[0].size(), _nonEmptyCellIDs.size());
}

void DenseBox::genReferencePoints() {
    for (auto emptyID : _nonEmptyCellIDs) {
        uint64 l0 = emptyID / _nCells[1];
        uint64 l1 = emptyID % _nCells[1];
    }
}

void DenseBox::fastDBSCAN() {
    auto fastDBSCAN = DenseDBSCAN(_epsilon, _minpts, _dataPoints);
    fastDBSCAN.run();
    _DBSCANCluster = fastDBSCAN.getCluster();
    auto _size = _DBSCANCluster.size();
    for (auto id = 0; id < _size; ++id) {
        _DBCluster[_densePointsID2dataPointsID[id]] = _curClusterNums + _DBSCANCluster[id];
    }
}

void DenseBox::reInterpretCluster() {
    set<int> curCluster;
    vector<int> clusterID, clusterIDMap;
    for (auto c : _DBCluster)
        curCluster.insert(c);
    for (auto c : curCluster)
        clusterID.push_back(c);
    std::sort(clusterID.begin(), clusterID.end());
    clusterIDMap.resize(clusterID.size());
    for (auto i = 0; i < clusterID.size(); ++i)
        clusterIDMap[clusterID[i]] = i;
    for (auto i = 0; i < _dataSize; ++i)
        _DBCluster[i] = clusterIDMap[_DBCluster[i]];
    _curClusterNums = curCluster.size();
}

bool cmp(const pair<int, int> a, const pair<int, int> b) {
    return a.second > b.second;
}

void DenseBox::reArrangeCluster() {
    vector<pair<int, int>> clusterID;
    vector<int> clusterIDMap;
    clusterID.resize(_curClusterNums);
    for (auto i = 0; i < _curClusterNums; ++i) {
        clusterID[i].first = i;
        clusterID[i].second = 0;
    }
    for (auto c : _DBCluster) {
        clusterID[c].second++;
    }
    std::sort(clusterID.begin(), clusterID.end(), cmp);
    // for (auto k : clusterID)
    //     fprintf(stdout, "(%d,%d )", k.first, k.second);
    clusterIDMap.resize(_curClusterNums);
    for (auto i = 0; i < _curClusterNums; ++i) {
        if (clusterIDMap[clusterID[i].second] > _minpts)
            clusterIDMap[clusterID[i].first] = i;
        else
            clusterIDMap[clusterID[i].first] = 0;
    }

    for (auto i = 0; i < _dataSize; ++i)
        _DBCluster[i] = clusterIDMap[_DBCluster[i]];
}

vector<int> DenseBox::getCellFromStepList(int& cellID, vector<int>& stepList) {
    vector<int> cells;
    for (int i = 0; i < stepList.size(); ++i) {
        auto _id = cellID + stepList[i];
        if (_id > -1 && _id < _totalCells) {
            cells.push_back(_id);
        }
    }
    return cells;
}

void DenseBox::mergeAdjacent() {
    auto stepList = vector<int>({-_nCells[1] - 1, -_nCells[1], -_nCells[1] + 1, -1, +1, _nCells[1] - 1, _nCells[1], _nCells[1] + 1});
    vector<bool> isVisited(_totalCells, false);
    for (auto i = 0; i < _totalCells; ++i) {
        if (_cellInners[i].size() && !isVisited[i]) {
            isVisited[i] = true;
            vector<int> bfsNeighbors;
            auto adjID = getCellFromStepList(i, stepList);
            for (auto _id : adjID) {
                bfsNeighbors.push_back(_id);
                isVisited[_id] = true;
            }
            for (auto nid = 0; nid < bfsNeighbors.size(); ++nid) {
                if (_cellInners[bfsNeighbors[nid]].size()) {
                    auto nextAdjs = getCellFromStepList(bfsNeighbors[nid], stepList);
                    for (auto _id : nextAdjs) {
                        if (isVisited[_id])
                            continue;
                        bfsNeighbors.push_back(_id);
                        isVisited[_id] = true;
                    }
                    for (auto innerPt : _cellInners[bfsNeighbors[nid]]) {
                        _DBCluster[innerPt] = i;
                    }
                    // fprintf(stdout, "merged adj %d to %d\n", bfsNeighbors[nid], i);
                }
            }
        }
    }
}

bool DenseBox::needMerge(int& cellID1, int& cellID2) {
    for (auto ptID1 : _cellInners[cellID1]) {
        for (auto ptID2 : _cellInners[cellID2]) {
            if (dist(ptID1, ptID2) < _oriEpsilon) return true;
        }
    }
    return false;
}

void DenseBox::mergeNonAdjacent() {
    auto stepList = vector<int>(
        {
            -2 * _nCells[1] - 1,
            -2 * _nCells[1],
            -2 * _nCells[1] + 1,
            -2 - _nCells[1],
            -2,
            -2 + _nCells[1],
            2 * _nCells[1] - 1,
            2 * _nCells[1],
            2 * _nCells[1] + 1,
            2 - _nCells[1],
            2,
            2 + _nCells[1],
        });
    vector<bool> isVisited(_totalCells, false);
    for (auto i = 0; i < _totalCells; ++i) {
        if (_cellInners[i].size() && !isVisited[i]) {
            isVisited[i] = true;
            vector<int> bfsNeighbors;
            auto adjID = getCellFromStepList(i, stepList);
            for (auto _id : adjID) {
                bfsNeighbors.push_back(_id);
                isVisited[_id] = true;
            }
            for (auto nid = 0; nid < bfsNeighbors.size(); ++nid) {
                if (_cellInners[bfsNeighbors[nid]].size() && needMerge(i, bfsNeighbors[nid])) {
                    auto nextAdjs = getCellFromStepList(bfsNeighbors[nid], stepList);
                    for (auto _id : nextAdjs) {
                        if (isVisited[_id])
                            continue;
                        bfsNeighbors.push_back(_id);
                        isVisited[_id] = true;
                    }
                    int con;
                    for (auto innerPt : _cellInners[bfsNeighbors[nid]]) {
                        con = _DBCluster[innerPt];
                        _DBCluster[innerPt] = i;
                    }
                    for (auto j = 0; j < _totalCells; ++j) {
                        if (_cellInners[j].size() && _DBCluster[_cellInners[j][0]] == con) {
                            for (auto innerPt : _cellInners[j]) {
                                _DBCluster[innerPt] = i;
                            }
                            isVisited[j] = true;
                        }
                    }
                    // fprintf(stdout, "merged non-adj %d to %d\n", bfsNeighbors[nid], i);
                }
            }
        }
    }
}

void DenseBox::mergeAll() {
    auto stepList = vector<int>(
        {
            -2 * _nCells[1] - 2,
            -2 * _nCells[1] + 2,
            2 * _nCells[1] - 2,
            2 * _nCells[1] + 2,
            -3 * _nCells[1] - 3,
            -3 * _nCells[1] - 2,
            -3 * _nCells[1] - 1,
            -3 * _nCells[1] - 0,
            -3 * _nCells[1] + 1,
            -3 * _nCells[1] + 2,
            -3 * _nCells[1] + 3,
            -3 - 2 * _nCells[1],
            3 - 2 * _nCells[1],
            -3 - _nCells[1],
            3 - _nCells[1],
            -3,
            3,
            -3 + _nCells[1],
            3 + _nCells[1],
            -3 + 2 * _nCells[1],
            3 + 2 * _nCells[1],
            3 * _nCells[1] - 3,
            3 * _nCells[1] - 2,
            3 * _nCells[1] - 1,
            3 * _nCells[1] - 0,
            3 * _nCells[1] + 1,
            3 * _nCells[1] + 2,
            3 * _nCells[1] + 3,
        });
    vector<bool> isVisited(_totalCells, false);
    for (auto i = 0; i < _totalCells; ++i) {
        if (_cellInners[i].size() && !isVisited[i]) {
            isVisited[i] = true;
            vector<int> bfsNeighbors;
            auto adjID = getCellFromStepList(i, stepList);
            for (auto _id : adjID) {
                bfsNeighbors.push_back(_id);
                isVisited[_id] = true;
            }
            for (auto nid = 0; nid < bfsNeighbors.size(); ++nid) {
                if (_cellInners[bfsNeighbors[nid]].size() && needMerge(i, bfsNeighbors[nid])) {
                    auto nextAdjs = getCellFromStepList(bfsNeighbors[nid], stepList);
                    for (auto _id : nextAdjs) {
                        if (isVisited[_id])
                            continue;
                        bfsNeighbors.push_back(_id);
                        isVisited[_id] = true;
                    }
                    int con;
                    for (auto innerPt : _cellInners[bfsNeighbors[nid]]) {
                        con = _DBCluster[innerPt];
                        _DBCluster[innerPt] = i;
                    }
                    for (auto j = 0; j < _totalCells; ++j) {
                        if (_cellInners[j].size() && _DBCluster[_cellInners[j][0]] == con) {
                            for (auto innerPt : _cellInners[j]) {
                                _DBCluster[innerPt] = i;
                            }
                            isVisited[j] = true;
                        }
                    }
                    // fprintf(stdout, "merged non-adj %d to %d\n", bfsNeighbors[nid], i);
                }
            }
        }
    }
}

void DenseBox::test() {
    clusterAndGetDensePoints();
    reInterpretCluster();
    fprintf(stdout, "After Dense Cluster, %d clusters\n", _curClusterNums);
    fastDBSCAN();
    reInterpretCluster();
    fprintf(stdout, "After DBSCAN Cluster, %d clusters\n", _curClusterNums);
    mergeAdjacent();
    reInterpretCluster();
    fprintf(stdout, "After Merge Adjacent, %d clusters\n", _curClusterNums);
    mergeNonAdjacent();
    reInterpretCluster();
    fprintf(stdout, "After Merge Non-Adjacent, %d clusters\n", _curClusterNums);
    mergeAll();
    reInterpretCluster();
    reArrangeCluster();
    reInterpretCluster();
    fprintf(stdout, "After Merge Reference Table, %d clusters\n", _curClusterNums);
}

float DenseBox::dist(const uint& id0, const uint& id1) {
    return pow(_dataPoints[0][id0] - _dataPoints[0][id1], 2) + pow(_dataPoints[1][id0] - _dataPoints[1][id1], 2);
}

vector<int> DenseBox::getCluster() {
    return _DBCluster;
}

DenseDBSCAN::DenseDBSCAN(
    const float epsilon,
    const uint minpts,
    DataPointsType dataPoints) : epsilon(epsilon),
                                 epsilonPow(pow(epsilon, 2)),
                                 minpts(minpts),
                                 dataPoints(dataPoints),
                                 dataSize(dataPoints[0].size()) {
    clusterIDs.resize(dataSize);
    for (int i = 0; i < dataSize; i++) {
        clusterIDs[i] = DenseDBSCAN::UNVISITED;
    }
}

void DenseDBSCAN::run() {
    // float start = omp_get_wtime();

    int clusterID = 0;
    for (uint i = 0; i < dataSize; i++) {
        if (clusterIDs[i] == DenseDBSCAN::UNVISITED) {
            auto neighbors = getNeighbors(i);
            if (neighbors.size() < minpts) {
                clusterIDs[i] = DenseDBSCAN::NOISE;
            } else {
                clusterIDs[i] = ++clusterID;
                for (uint j = 0; j < neighbors.size(); j++) {
                    // printf("j = %u, size = %d\n", j, neighbors.size());
                    uint p = neighbors[j];
                    if (clusterIDs[p] == DenseDBSCAN::UNVISITED) {
                        auto pNeighbors = getNeighbors(p);
                        if (pNeighbors.size() >= minpts) {
                            neighbors.insert(neighbors.end(), pNeighbors.begin(), pNeighbors.end());
                        }
                    }
                    if (clusterIDs[p] == DenseDBSCAN::UNVISITED || clusterIDs[p] == DenseDBSCAN::NOISE) {
                        clusterIDs[p] = clusterID;
                    }
                }
            }
        }
    }
    // float end = omp_get_wtime();
    // float elapsed_time_ms = (end - start) * 1000;
    // printf("Time elapsed on original DBSCAN: %f ms\n", elapsed_time_ms);
}

float DenseDBSCAN::dist(const uint& id0, const uint& id1) {
    return pow(dataPoints[0][id0] - dataPoints[0][id1], 2) + pow(dataPoints[1][id0] - dataPoints[1][id1], 2);
}

vector<uint> DenseDBSCAN::getNeighbors(const uint& id) {
    //TODO: Change to get neighbors from neighborTable
    auto neighbors = vector<uint>();
    neighbors.clear();
    for (uint i = 0; i < dataSize; i++) {
        if (DenseDBSCAN::dist(id, i) <= epsilonPow) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

vector<int>& DenseDBSCAN::getCluster() {
    return clusterIDs;
}