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
        double DBfraction=1.0;
        unsigned int stride=1.0/DBfraction;

        //histogram the data in the first dimension
        unsigned long int *bins=new unsigned long int [nCells[0]];

        //find the Bin IDs with the cumulative fraction
        double *cumulativeFrac=new double [nCells[0]];

        for (unsigned int i=0; i<nCells[0]; i++)
        {
            cumulativeFrac[i]=0;
            bins[i]=0;
        }

        unsigned int binidx=0;
        unsigned int approxTotalPointsInserted=(dataPoints[0].size()*1.0)*DBfraction;
        // printf("\nApprox points inserted: %u", approxTotalPointsInserted);

        for (unsigned long int i=0; i<dataPoints[0].size(); i+=stride)
        {
            binidx=(dataPoints[0][i]-minVals[0])/epsilon;
            bins[binidx]++;
        }

        
        

        //the first cumulative fraction is the number of points in the bim
        cumulativeFrac[0]=bins[0];
        //get the number
        for (unsigned long int i=1; i<nCells[0]; i++)
        {
            cumulativeFrac[i]=bins[i]+cumulativeFrac[i-1];
            // printf("\nbin: %lu, cumulative frac (num): %f",i,cumulativeFrac[i]);
        }

        //convert to fraction
        for (unsigned long int i=0; i<nCells[0]; i++)
        {
            cumulativeFrac[i]=cumulativeFrac[i]/(approxTotalPointsInserted*1.0);
            // printf("\nbin: %lu, cumulative frac (fraction): %f",i,cumulativeFrac[i]);
        }

        //find bin boundaries
        double fractionDataPerPartition=1.0/(NCHUNKS*1.0);
        double boundary=fractionDataPerPartition;
        // printf("\nFrac data per partition: %f", fractionDataPerPartition);

        binBounaries[0]=0;
        unsigned int cntBin=1;
        for (unsigned long int i=0; i<nCells[0]-1; i++)
        {
            
            // printf("\nBoundary: %f", boundary);
            // printf("\ni: %lu, cntbin: %u",i,cntBin);
            if (boundary>=cumulativeFrac[i] && boundary<cumulativeFrac[i+1])
            {
                binBounaries[cntBin]=i;
                cntBin++;
                boundary+=fractionDataPerPartition;
            }

                
        }


        binBounaries[NCHUNKS]=nCells[0];

        // for (unsigned int i=0; i<CHUNKS+1; i++)
        // {
        // 	printf("\nBin boundaries (cell of total num cells): %u/%u",binBounaries[i],nCells[0]);
        // }	


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
	
        //compute the binids that are in the shadow region
        if (CHUNKS>1)
        {
            //left-most partition doesn't have a region on the left grid edge
            shadowBins.push_back(binBounaries[1]-1);
            shadowBins.push_back(binBounaries[1]);
            
            

            //middle partitions
            for (unsigned int i=1; i<CHUNKS-1; i++)
            {
                //add left shadow regions
                shadowBins.push_back(binBounaries[i]+1);
                shadowBins.push_back(binBounaries[i]+2);

                //right shadow region
                shadowBins.push_back(binBounaries[i+1]-1);
                shadowBins.push_back(binBounaries[i+1]);
            }


            //right-most partition doesn't have a region on the right grid edge
            shadowBins.push_back(binBounaries[CHUNKS-1]+1);
            shadowBins.push_back(binBounaries[CHUNKS-1]+2);

            // for (unsigned int i=0; i<shadowBins.size(); i++)
            // {
            // 	printf("\nShadow region cell id: %u",shadowBins[i] );
            // }


            //check to make sure that the shadow regions are disjoint
            //i.e., no two identical bins in the shadow region
            //if not we need to quit

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
        
        
        //add points to shadow region
        for (unsigned int i=0; i<dataPoints[0].size(); i++)
        {
            
            unsigned int binidx=(dataPoints[0][i]-minVals[0])/epsilon;
            //add points to the shadow region if it falls in the region
            //These are cells in two cells on each border on the left and right of
            //each partition, except the left and right-most partitions
            auto it = std::lower_bound(shadowBins.begin(), shadowBins.end(),binidx);
            if(!(it == shadowBins.end() || *it != binidx))
            {
                pointIDs_shadow.push_back(i);
                for (int j=0; j<2; j++)
                {
                    // NDdataPointsInShadowRegion[k][i].push_back((*NDdataPoints)[k][i]);
                    // (*NDdataPointsInShadowRegion)[k][i]=(*NDdataPoints)[k][i];
                    dataPoints_shadow[j].push_back(dataPoints[j][i]);

                }
                

                
            }
        }
        
        

        //Step1: for each point, compute its partition -- can be done in parallel
        std::vector<unsigned int> mapPointToPartition; 
        mapPointToPartition.resize(dataPoints[0].size());

        
        

        //Some threads may get lots of inner loop iterations, need to address load imbalance
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
        //Num threads is either the chunks or the num threads, whichever is lower
        const unsigned int NTHREADSPARTITION=min(NUM_THREADS,CHUNKS);
        std::vector<unsigned int> prefixSumPartition[CHUNKS];
        std::vector<unsigned int> pointIDsPartition[CHUNKS];
        unsigned int prefixSums[CHUNKS]; 

        for (unsigned int i=0; i<CHUNKS; i++)
        {
            prefixSums[i]=0;
        }

        //prefix sum for each CHUNK
        for (unsigned int i=0; i<dataPoints[0].size(); i++)
        {
            unsigned int partition=mapPointToPartition[i];
            prefixSumPartition[partition].push_back(prefixSums[partition]);
            pointIDsPartition[partition].push_back(i);
            prefixSums[partition]++;
        }



        

        //resize arrays
        for (unsigned int i=0; i<CHUNKS; i++){
            for (int j=0; j<2; j++){
            partDataPointsArray[i][j].resize(pointIDsPartition[i].size());
            }
        }
        pointChunkMapping.resize(dataPoints[0].size());

        

        
        //write the results using prefix sums in parallel

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

                //update mapping for the point in the entire (global) dataset
                PointChunkLookup tmp;
                tmp.pointID=dataidx;
                tmp.chunkID=i;
                tmp.idxInChunk=idx;
                pointChunkMapping[dataidx]=tmp;	
            }

        }
}