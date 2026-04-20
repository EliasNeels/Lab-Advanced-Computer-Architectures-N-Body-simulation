#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"

struct OctNode {
    float center_x, center_y, center_z;  // octant center
    float size;                           // octant half-width
    float com_x, com_y, com_z;           // center of mass
    float total_mass;                     // total mass in this node
    int   children;                       // index of first child (0 = leaf)
    int   body_start;                     // start index in sorted body array
    int   body_count;                     // number of bodies in this node
    int   next;                           // next node for stackless traversal
};

struct Octree {
    OctNode* nodes;
    int* parents;
    int* nodeCount;
    int maxNodes;
};