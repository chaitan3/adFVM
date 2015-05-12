#include <stdio.h>
#define Ne 100000
#define Ns 40000

int main() {
    int a[Ne], b[Ns], c[Ns];
    int i, j;
    for (i=0; i < Ns; i++) {
        b[i] = 2*i;
    }
    for (j=0; j < 10000; j++) {
        #pragma omp parallel for
        for (i=0; i < Ns; i++) {
            c[i] = a[b[i]];
        }
    }

    return 0;
}
