#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <set>


long *CppMapLabels(long *segmentation, long *mapping, unsigned long nentries)
{
  long *updated_segmentation = new long[nentries];
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    updated_segmentation[iv] = mapping[segmentation[iv]];
  }
  return updated_segmentation;
}

long *CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries)
{
  if (threshold == 0) return segmentation;
  /* TODO can I assume that there are an integer number of voxels */
  // find the maximum label
  long max_segment_label = 0;
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (segmentation[iv] > max_segment_label) max_segment_label = segmentation[iv];
  }
  max_segment_label++;
  
  // create a counter array for the number of voxels
  int *nvoxels_per_segment = new int[max_segment_label];
  for (long iv = 0; iv < max_segment_label; ++iv) {
    nvoxels_per_segment[iv] = 0;
  }
  // count the number of voxels per segment
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    nvoxels_per_segment[segmentation[iv]]++;
  }
  // create the array for the updated segmentation
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (nvoxels_per_segment[segmentation[iv]] < threshold) segmentation[iv] = 0;
  }
  // free memory
  delete[] nvoxels_per_segment;
  return segmentation;
}

static unsigned long row_size;
static unsigned long sheet_size;

void IndexToIndicies(long iv, long &ix, long &iy, long &iz)
{
  iz = iv / sheet_size;
  iy = (iv - iz * sheet_size) / row_size;
  ix = iv % row_size;
}


long IndiciesToIndex(long ix, long iy, long iz)
{
  return iz * sheet_size + iy * row_size + ix;
}

long *CppForceConnectivity(long *segmentation, long zres, long yres, long xres)
{
  // create the new components array
  unsigned long nentries = zres * yres * xres;
  long *components = new long[nentries];
  for (unsigned long iv = 0; iv < nentries; ++iv)
    components[iv] = 0;

  // set global variables
  row_size = xres;
  sheet_size = yres * xres;


  // create the queue of labels
  std::queue<unsigned long> pixels = std::queue<unsigned long>();

 unsigned long current_index = 0;
 unsigned long current_label = 1;

  while (current_index < nentries) {
    // set this component and add to the queue
    components[current_index] = current_label;
    pixels.push(current_index);

    // iterate over all pixels in the queue
    while (pixels.size()) {
      // remove this pixel from the queue
      unsigned long pixel = pixels.front();
      pixels.pop();
 
      // add the six neighbors to the queue
      long iz, iy, ix;
      IndexToIndicies(pixel, ix, iy, iz);

      for (long iw = -1; iw <= 1; ++iw) {
        if (iz + iw < 0 or iz + iw > zres - 1) continue;
        for (long iv = -1; iv <= 1; ++iv) {
          if (iy + iv < 0 or iy + iv > yres - 1) continue;
          for (long iu = -1; iu <= 1; ++iu) {
            if (ix + iu < 0 or ix + iu > xres - 1) continue;
            long neighbor = IndiciesToIndex(ix + iu, iy + iv, iz + iw);
            if (segmentation[pixel] == segmentation[neighbor] && !components[neighbor]) {
              components[neighbor] = current_label;
              pixels.push(neighbor);
            }
          }
        }
      }
    }
    current_label++;

    // if the current index is already labeled, continue
    while (current_index < nentries && components[current_index]) current_index++;
  }

  // create a list of mappings
  unsigned long max_segment = 0;
  unsigned long max_component = 0;
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (segmentation[iv] > max_segment) max_segment = segmentation[iv];
    if (components[iv] > max_component) max_component = components[iv];
  }
  max_segment++;
  max_component++;

  std::set<long> *seg2comp = new std::set<long>[max_segment];
  for (unsigned long iv = 0; iv < max_segment; ++iv)
    seg2comp[iv] = std::set<long>();

  // see if any segments have multiple components
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    seg2comp[segmentation[iv]].insert(components[iv]);
  }

  long overflow = max_segment;
  long *comp2seg = new long[max_component];
  for (unsigned long iv = 1; iv < max_segment; ++iv) {
    if (seg2comp[iv].size() == 1) {
      // get the component for this segment
      long component = *(seg2comp[iv].begin());
      comp2seg[component] = iv;
    }
    else {
      // iterate over the set
      for (std::set<long>::iterator it = seg2comp[iv].begin(); it != seg2comp[iv].end(); ++it) {
        long component = *it;

        // one of the components keeps the label
        if (it == seg2comp[iv].begin()) comp2seg[component] = iv;
        // set the component to start at max_segment and increment
        else {
          comp2seg[component] = overflow;
          ++overflow;
        }
      }
    }
  }

  // update the segmentation
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (!segmentation[iv]) components[iv] = 0;
    else components[iv] = comp2seg[components[iv]];
  }

  // free memory
  delete[] seg2comp;
  delete[] comp2seg;

  return components;
}


