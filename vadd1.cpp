#include "util.hpp"

int LENGTH = 1000000;

extern double wtime();       // returns time since some fixed past point (wtime.c)

int main(int argc, char** argv) {

    util::Timer timer;

    float* h_a = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float* h_b = (float*) calloc(LENGTH, sizeof(float));       // b vector
    float* h_c = (float*) calloc(LENGTH, sizeof(float));       // c vector (a+b) returned from the compute device
    
    // Fill vectors a and b with random float values
    int count = LENGTH;
    for (int i = 0; i < 1000; i++) {
        for (int i = 0; i < count; i++) {
            h_a[i] = rand() / (float)RAND_MAX;
            h_b[i] = rand() / (float)RAND_MAX;
        }


        for (int i = 0; i < count; i++) {
            h_c[i] = h_a[i] + h_b[i];
        }
    }

    double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
    printf("\nThe program ran in %lf seconds\n", rtime);

    return 0;
}

