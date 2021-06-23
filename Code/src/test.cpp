#include <omp.h>

#include <iostream>
using namespace std;

#define GPU_STREAMS 3

int main() {
#pragma omp parallel for schedule(static, 1) num_threads(GPU_STREAMS)
    for (int i = 0; i < 20; i++) {
        printf("i = %d, tid = %d\n", i, omp_get_thread_num());
    }
    return 0;
}