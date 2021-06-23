#include "Dataset.hpp"

std::vector<float> Dataset::__naiveLoadFromTxt(const std::string txtPath) {
  std::vector<float> txtData;
  std::ifstream txtFile(txtPath);
  if (!txtFile) {
    fprintf(stdout, "Open txt File failed\n");
  }

  for (std::string f; getline(txtFile, f, ',');) {
    float i;
    std::stringstream ss(f);
    while (ss >> i) {
      txtData.push_back(i);
      if (ss.peek() == ',') ss.ignore();
    }
  }

  txtFile.close();
  return txtData;
}

int Dataset::loadFromBin(const std::string fromBin) {
  std::fstream binFile(fromBin, std::ios::in | std::ios::binary);
  if (!binFile) {
    fprintf(stdout, "Open Input File failed\n");
    return 0;
  }
  std::vector<float> x, y;
  float tmp;
  bool flag = false;
  while (binFile.read(reinterpret_cast<char*>(&tmp), sizeof(float))) {
    if (flag)
      _points[1].push_back(tmp);
    else
      _points[0].push_back(tmp);
    flag = !flag;
  }
  _size = _points[0].size();
  fprintf(stdout, "Loaded %d floats\n", _size * 2);
  return 0;
}

int Dataset::loadFromTxt(const std::string fromTxt) {
  std::ifstream in(fromTxt);
  std::string str;
  unsigned idx = 0;
  float x;

  if (in.is_open()) {
    while (getline(in, str, ',')) {
      std::stringstream ss(str);
      while (ss >> x) {
        _points[idx].push_back(x);
        idx = 1 - idx;
      }
    }
    in.close();
  }
  _size = _points[0].size();
  return 0;
}

int Dataset::genDatasetFromTxt(const std::string fromTxt,
                               const std::string toBin) {
  auto txtData = __naiveLoadFromTxt(fromTxt);
  std::fstream binFile(toBin, std::ios::out | std::ios::binary);
  if (!binFile) {
    fprintf(stdout, "Open Output File failed\n");
    return 0;
  }
  for (auto i : txtData)
    binFile.write(reinterpret_cast<char*>(&i), sizeof(float));
  return 0;
}

void Dataset::loadDataset(const std::string file) {
  auto ext = file.substr(file.length() - 3, 3);
  fprintf(stdout, "File Ext: %s\n", ext.c_str());
  _points[0].clear();
  _points[1].clear();
  if (ext == "dat")
    loadFromBin(file);
  else if (ext == "txt")
    loadFromTxt(file);
  else
    fprintf(stderr, "Invalid File Ext\n");
}

void Dataset::showDataset(int head) {
  for (auto i = 0; i < head; ++i) {
    if (i < _size) {
      fprintf(stdout, "x: %.10f, y: %.10f\n", _points[0][i], _points[1][i]);
    } else
      break;
  }
}

const DataPointsType& Dataset::getDataset() { return _points; }
