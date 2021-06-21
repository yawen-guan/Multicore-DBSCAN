#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>

#define min(a,b)    (((a) < (b)) ? (a) : (b))

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::stringstream;

DataPointsType importDataset(const string datafile) {
    DataPointsType dataPoints;
    ifstream in(datafile);
    string str;
    uint idx = 0;
    float x;

    if (in.is_open()) {
        while (getline(in, str, ',')) {
            stringstream ss(str);
            while (ss >> x) {
                dataPoints[idx].push_back(x);
                idx = 1 - idx;
            }
            // ;
            // cout << "str = " << str << " x = " << x << endl;
        }
        in.close();
    }

    return dataPoints;
}

void generateGridDimensions(
    const DataPointsType &dataPoints,
    const float &epsilon,
    MinMaxValsType &minVals,
    MinMaxValsType &maxVals,
    array<uint, 2> &nCells,
    uint64 &totalCells) {
        if(dataPoints[0].size() == 0){
            printf("Empty dataPoints !\n");
            return;
        }
        // the minimum/maximum value of the points in each dimensions +/- epsilon
        // add buffer around each dim so no weirdness later with putting data into cells
        for (int j=0; j<2; j++){
            minVals[j]=dataPoints[j][0];
            maxVals[j]=dataPoints[j][0];
        }

        for (int i=1; i<dataPoints[0].size(); i++){
            for (int j=0; j<2; j++){
            if (dataPoints[j][i]<minVals[j]){
                minVals[j]=dataPoints[j][i];
            }
            if (dataPoints[j][i]>maxVals[j]){
                maxVals[j]=dataPoints[j][i];
            }	
            }
        }

        for (int j=0; j<2; j++){
            minVals[j]-=epsilon;
            maxVals[j]+=epsilon;
        }	

        
        //find totalCells
        for (int j=0; j<2; j++){
            nCells[j]=ceil((maxVals[j]-minVals[j])/epsilon);
        }

        totalCells = nCells[0]*nCells[1];


}

void generatePartitions(
    const DataPointsType &dataPoints,
    const float &epsilon,
    const MinMaxValsType &minVals,
    const MinMaxValsType &maxVals,
    const array<uint, 2> &nCells,
    const uint64 &totalCells,
    vector<uint> &binBounaries,
    const int &NCHUNKS) {

        //find binBounaries of cells

        double DBfraction=1.0;
        unsigned int stride=1.0/DBfraction;
        unsigned long int *bins=new unsigned long int [nCells[0]];
        double *cumulativeFrac=new double [nCells[0]];
        for (unsigned int i=0; i<nCells[0]; i++){
            cumulativeFrac[i]=0;
            bins[i]=0;
        }

        unsigned int binidx=0;
        unsigned int approxTotalPointsInserted=(dataPoints[0].size()*1.0)*DBfraction;
        
        // sum the data
        for (unsigned long int i=0; i<dataPoints[0].size(); i+=stride){
            binidx=(dataPoints[0][i]-minVals[0])/epsilon;
            bins[binidx]++;
        }

        
        // calu cumulativeFrac
        cumulativeFrac[0]=bins[0];
        for (unsigned long int i=1; i<nCells[0]; i++){
            cumulativeFrac[i]=bins[i]+cumulativeFrac[i-1];
        }
        for (unsigned long int i=0; i<nCells[0]; i++){
            cumulativeFrac[i]=cumulativeFrac[i]/(approxTotalPointsInserted*1.0);
        }

        //find bin boundaries
        double fractionDataPerPartition=1.0/(NCHUNKS*1.0);
        double boundary=fractionDataPerPartition;
        
        // calu binBounaries of cells
        binBounaries[0]=0;
        unsigned int cntBin=1;
        for (unsigned long int i=0; i<nCells[0]-1; i++){
            if (boundary>=cumulativeFrac[i] && boundary<cumulativeFrac[i+1]){
                binBounaries[cntBin]=i;
                cntBin++;
                boundary+=fractionDataPerPartition;
            }  
        }
        binBounaries[NCHUNKS]=nCells[0];

        delete [] bins;
        delete [] cumulativeFrac;

}

void generatePartitionDatasets(
    const DataPointsType &dataPoints,
    const float &epsilon,
    const MinMaxValsType &minVals,
    const MinMaxValsType &maxVals,
    const vector<uint> &binBounaries,
    const array<uint, 2> &nCells,
    vector<DataPointsType> &partDataPointsArray,
    vector<PointChunkLookup> &pointChunkMapping,
    vector<uint> &pointIDs_shadow,
    DataPointsType &dataPoints_shadow) {
        std::vector<unsigned int> shadowBins;

        const unsigned int CHUNKS = binBounaries.size()-1;
	
        //finf bin in the shadow
        if (CHUNKS>1){
            shadowBins.push_back(binBounaries[1]-1);
            shadowBins.push_back(binBounaries[1]);
            for (unsigned int i=1; i<CHUNKS-1; i++){
                shadowBins.push_back(binBounaries[i]+1);
                shadowBins.push_back(binBounaries[i]+2);

                shadowBins.push_back(binBounaries[i+1]-1);
                shadowBins.push_back(binBounaries[i+1]);
            }
            shadowBins.push_back(binBounaries[CHUNKS-1]+1);
            shadowBins.push_back(binBounaries[CHUNKS-1]+2);

            bool shadow_disjoint_flag=false;
            for (unsigned int i=1; i<shadowBins.size(); i++)
            {
                if (shadowBins[i]==shadowBins[i-1])
                {
                    shadow_disjoint_flag=true;
                    printf("\nOverlap in the shadow region!! At least two bins are the same.");
                }
            }

            assert(shadow_disjoint_flag==false);

        }

        //add shadow points
        for (unsigned int i=0; i<dataPoints[0].size(); i++){
            
            unsigned int binidx=(dataPoints[0][i]-minVals[0])/epsilon;
            auto it = std::lower_bound(shadowBins.begin(), shadowBins.end(),binidx);
            if(!(it == shadowBins.end() || *it != binidx)){
                pointIDs_shadow.push_back(i);
                for (int j=0; j<2; j++){
                    dataPoints_shadow[j].push_back(dataPoints[j][i]);
                }
            }
        }
        
        

        //Step1
        std::vector<unsigned int> mapPointToPartition; 
        mapPointToPartition.resize(dataPoints[0].size());

        
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, 1)
        for (unsigned int i=0; i<dataPoints[0].size(); i++)
        {
                unsigned int binidx=(dataPoints[0][i]-minVals[0])/epsilon;
                for (unsigned int j=0; j<CHUNKS+1; j++){
                    if (binidx>=binBounaries[j] && binidx<binBounaries[j+1]){
                        mapPointToPartition[i]=j;
                        break;
                    }		
                }

        }


        

        //step2: 
        // const unsigned int NTHREADSPARTITION=min(NUM_THREADS,CHUNKS);
        std::vector<unsigned int> prefixSumPartition[CHUNKS];
        std::vector<unsigned int> pointIDsPartition[CHUNKS];
        unsigned int prefixSums[CHUNKS]; 

        for (unsigned int i=0; i<CHUNKS; i++)
        {
            prefixSums[i]=0;
        }

        //prefix sum
        for (unsigned int i=0; i<dataPoints[0].size(); i++)
        {
            unsigned int partition=mapPointToPartition[i];
            prefixSumPartition[partition].push_back(prefixSums[partition]);
            pointIDsPartition[partition].push_back(i);
            prefixSums[partition]++;
        }



        

        //resize
        partDataPointsArray.resize(CHUNKS);
        for (unsigned int i=0; i<CHUNKS; i++){
            
            for (int j=0; j<2; j++){
            partDataPointsArray[i][j].resize(pointIDsPartition[i].size());
            }
        }
        pointChunkMapping.resize(dataPoints[0].size());

        

        
        //make final partition
        for (unsigned int i=0; i<CHUNKS; i++)
        {
            #pragma omp parallel for num_threads(NUM_THREADS) shared(i)
            for (unsigned int j=0; j<pointIDsPartition[i].size(); j++)
            {
                unsigned int idx=prefixSumPartition[i][j];
                unsigned int dataidx=pointIDsPartition[i][j];
                
                for (int j=0; j<2; j++)
                {
                partDataPointsArray[i][j][idx]=dataPoints[j][dataidx];	
                }

                
                PointChunkLookup tmp;
                tmp.pointID=dataidx;
                tmp.chunkID=i;
                tmp.idxInChunk=idx;
                pointChunkMapping[dataidx]=tmp;	
            }

        }
}