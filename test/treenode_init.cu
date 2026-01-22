#ifdef COLLISION

#include "cudust_kern.cuh"

// =========================================================================================================================
// Kernel: treenode_init
// Purpose: Initialize KD-tree nodes from particle positions (spherical to Cartesian conversion)
// Dependencies: tree structure, swarm structure
// =========================================================================================================================

__global__
void treenode_init (tree *dev_treenode, const swarm *dev_particle)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;

    if (idx < N_PAR)
    {
        float x = static_cast<float>(dev_particle[idx].position.x);
        float y = static_cast<float>(dev_particle[idx].position.y);
        float z = static_cast<float>(dev_particle[idx].position.z);

        dev_treenode[idx].cartesian.x = y*sin(z)*cos(x);
        dev_treenode[idx].cartesian.y = y*sin(z)*sin(x);
        dev_treenode[idx].cartesian.z = y*cos(z);
        dev_treenode[idx].index_old   = idx;
    }
}

#endif // COLLISION
