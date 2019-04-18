#include <math.h>
#include <ctime>
#include <stdio.h>
#include <vector>
#include <stdlib.h>

// constant variables

static const int IB_Z = 0;
static const int IB_Y = 1;
static const int IB_X = 2;



// global variables

static long nentries;
static long sheet_size;
static long row_size;
static long infinity;



// helper functions

static long IndicesToIndex(long ix, long iy, long iz)
{
    return iz * sheet_size + iy * row_size + ix;
}



float *CppTwoDimensionalDistanceTransform(long *segmentation, long resolution[3])
{
    // initialize convenient variables for distances
    nentries = resolution[IB_Z] * resolution[IB_Y] * resolution[IB_X];
    sheet_size = resolution[IB_Y] * resolution[IB_X];
    row_size = resolution[IB_X];
    infinity = resolution[IB_Z] * resolution[IB_Z] + resolution[IB_Y] * resolution[IB_Y] + resolution[IB_X] * resolution[IB_X];

    float *DBF = new float[nentries];

    // allocate memory for bounday map and distance transform
    float *b = new float[nentries];
    for (long iz = 0; iz < resolution[IB_Z]; ++iz) {
        for (long iy = 0; iy < resolution[IB_Y]; ++iy) {
            for (long ix = 0; ix < resolution[IB_X]; ++ix) {
                long label = segmentation[IndicesToIndex(ix, iy, iz)];

                if ((ix > 0 and segmentation[IndicesToIndex(ix - 1, iy, iz)] != label) ||
                    (iy > 0 and segmentation[IndicesToIndex(ix, iy - 1, iz)] != label) ||
                    (ix < resolution[IB_X] - 1 and segmentation[IndicesToIndex(ix + 1, iy, iz)] != label) ||
                    (iy < resolution[IB_Y] - 1 and segmentation[IndicesToIndex(ix, iy + 1, iz)] != label)) {
                    b[IndicesToIndex(ix, iy, iz)] = 0;
                }
                else {
                    b[IndicesToIndex(ix, iy, iz)] = infinity;
                }
            }
        }
    }

    // go along the y dimension second for every (z, x) coordinate
    for (long iz = 0; iz < resolution[IB_Z]; ++iz) {
        for (long ix = 0; ix < resolution[IB_X]; ++ix) {

            long k = 0;
            long *v = new long[resolution[IB_Y]];
            float *z = new float[resolution[IB_Y]];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < resolution[IB_Y]; ++q) {
                // label for jump statement
                ylabel:
                float s = ((b[IndicesToIndex(ix, q, iz)] + q * q) - (b[IndicesToIndex(ix, v[k], iz)] + v[k] * v[k])) / (float)(2 * q - 2 * v[k]);

                if (s <= z[k]) {
                    --k;
                    goto ylabel;
                }
                else {
                    ++k; 
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0; q < resolution[IB_Y]; ++q) {
                while (z[k + 1] < q)
                    ++k;
                DBF[IndicesToIndex(ix, q, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(ix, v[k], iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    // update the boundary values with this distance
    for (long iz = 0; iz < resolution[IB_Z]; ++iz) {
        for (long iy = 0; iy < resolution[IB_Y]; ++iy) {
            for (long ix = 0; ix < resolution[IB_X]; ++ix) {
                b[IndicesToIndex(ix, iy, iz)] = DBF[IndicesToIndex(ix, iy, iz)];
            }
        }
    }



    // go along the x dimension last for every (y, z) coordinate
    for (long iy = 0; iy < resolution[IB_Y]; ++iy) {
        for (long iz = 0; iz < resolution[IB_Z]; ++iz) {

            long k = 0;
            long *v = new long[resolution[IB_X]];
            float *z = new float[resolution[IB_X]];

            v[0] = 0;
            z[0] = -1 * infinity;
            z[1] = infinity;

            for (long q = 1; q < resolution[IB_X]; ++q) {
                // label for jump statement
                xlabel:
                float s = ((b[IndicesToIndex(q, iy, iz)] + q * q) - (b[IndicesToIndex(v[k], iy, iz)] +  v[k] * v[k])) / (float)(2 * q - 2 * v[k]);

                if (s <= z[k]) {
                    --k;
                    goto xlabel;
                }
                else {
                    ++k;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = infinity;
                }
            }

            k = 0;
            for (long q = 0;  q < resolution[IB_X]; ++q) {
                while (z[k + 1] < q)
                    ++k;

                DBF[IndicesToIndex(q, iy, iz)] = (q - v[k]) * (q - v[k]) + b[IndicesToIndex(v[k], iy, iz)];
            }

            // free memory
            delete[] v;
            delete[] z;
        }
    }

    // free memory
    delete[] b;

    // update the distances to be euclidean instead of euclidean squared
    for (long iv = 0; iv < nentries; ++iv)
        DBF[iv] = sqrt(DBF[iv]);

    return DBF;
}


long *CppDilateData(long *data, long resolution[3], float distance)
{
    // initialize convenient variables for distances
    nentries = resolution[IB_Z] * resolution[IB_Y] * resolution[IB_X];
    sheet_size = resolution[IB_Y] * resolution[IB_X];
    row_size = resolution[IB_X];

    float *distances = CppTwoDimensionalDistanceTransform(data, resolution);

    // mask out distances that are two close
    long *masked_data = new long[nentries];
    for (long iv = 0; iv < nentries; ++iv) {
        if (distances[iv] <= distance) masked_data[iv] = 0;
        else masked_data[iv] = data[iv];
    }

    // free memory
    delete[] distances;

    return masked_data;
}