#pragma once
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "global.hpp"

class Dataset {
private:
    // Private Variables
    DataPointsType _points;
    unsigned _size;

    // Private Functions
    std::vector<float> __naiveLoadFromTxt(const std::string txtPath);
    int loadFromBin(const std::string fromBin);
    int loadFromTxt(const std::string fromTxt);

public:
    int genDatasetFromTxt(const std::string fromTxt, const std::string toBin);
    void loadDataset(const std::string file);
    void showDataset(int head = 100);

    // Get Funtions
    const DataPointsType& getDataset();
};