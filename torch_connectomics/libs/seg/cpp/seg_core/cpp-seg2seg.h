long *CppMapLabels(long *segmentation, long *mapping, unsigned long nentries);
long *CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries);
long *CppForceConnectivity(long *segmentation, long zres, long yres, long xres);