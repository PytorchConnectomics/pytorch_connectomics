#include <stdio.h>
#include "cpp-seg2gold.h"
#include <map>


long *CppMapping(long *segmentation, int *gold, long nentries, double low_threshold, double high_threshold)
{
    // find the maximum segmentation value
    long max_segmentation_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (segmentation[iv] > max_segmentation_value)
            max_segmentation_value = segmentation[iv];
    }
    max_segmentation_value++;

    // find the maximum gold value
    long max_gold_value = 0;
    for (long iv = 0; iv < nentries; ++iv) {
        if (gold[iv] > max_gold_value) 
            max_gold_value = gold[iv];
    }
    max_gold_value++;

    // find the number of voxels per segment
    unsigned long *nvoxels_per_segment = new unsigned long[max_segmentation_value];
    for (long iv = 0; iv < max_segmentation_value; ++iv)
        nvoxels_per_segment[iv] = 0;

    /* TODO way too memory expensive */
    unsigned long **seg2gold_overlap = new unsigned long *[max_segmentation_value];
    for (unsigned long is = 0; is < max_segmentation_value; ++is) {
        seg2gold_overlap[is] = new unsigned long[max_gold_value];
        for (unsigned long ig = 0; ig < max_gold_value; ++ig) {
            seg2gold_overlap[is][ig] = 0;
        }
    }

    // iterate over every voxel
    for (long iv = 0; iv < nentries; ++iv) {
        nvoxels_per_segment[segmentation[iv]]++;
        seg2gold_overlap[segmentation[iv]][gold[iv]]++;
    }

    // create the mapping
    long *segmentation_to_gold = new long[max_segmentation_value];
    for (long is = 0; is < max_segmentation_value; ++is) {
        long gold_id = 0;
        long gold_max_value = 0;

        // only gets label of 0 if the number of non zero voxels is below threshold
        for (long ig = 1; ig < max_gold_value; ++ig) {
            if (seg2gold_overlap[is][ig] > gold_max_value) {
                gold_max_value = seg2gold_overlap[is][ig];
                gold_id = ig;
            }
        }

        // the number of non zeros pixels must be greater than 10%
        if (gold_max_value / (double)nvoxels_per_segment[is] < low_threshold) segmentation_to_gold[is] = 0;
        // number of non zero pixels must be greater than threhsold
        else if (gold_max_value / (double)(nvoxels_per_segment[is] - seg2gold_overlap[is][0]) > high_threshold) segmentation_to_gold[is] = gold_id;
        else segmentation_to_gold[is] = 0;
    }

    // free memory
    for (long is = 0; is < max_segmentation_value; ++is) {
        delete[] seg2gold_overlap[is];
    }
    delete[] seg2gold_overlap;
    delete[] nvoxels_per_segment;

    return segmentation_to_gold;
}