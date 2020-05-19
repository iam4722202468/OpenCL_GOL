__kernel void GOL(const int dim,
                  __global int *grid,
                  __global int *gridSwp)
{
    __private int x = get_global_id(0);
    __private int y = get_global_id(1);

    __private int id = x * dim + y;
    __private int max = dim*dim;

    __private int neighborCount = 0;
    
    // Positions of neighbors
    __private int positions[8] = {id-dim-1, id+dim-1, id-1, id+dim, id-dim, id+1, id-dim+1, id+dim+1};

    // If we're on the left edge don't do left up, left or left down
    __private int leftEdge = 3 * (id % dim == 0);

    // If we're on the right edge don't do right up, right or right down
    __private int rightEdge = 3 * ((id + 1) % dim == 0);

    // Go through neighbors and check sum
    for (__private int i = leftEdge; i < 8 - rightEdge; ++i)
        if (positions[i] >= 0 && positions[i] < max) // Don't go out of bounds
            neighborCount += grid[positions[i]];

    __private int current = grid[id];

    if (current == 1 && neighborCount < 2)
        gridSwp[id] = 0;
    else if (current == 1 && (neighborCount == 2 || neighborCount == 3))
        gridSwp[id] = 1;
    else if (current == 1 && neighborCount > 3)
        gridSwp[id] = 0;
    else if (current == 0 && neighborCount == 3)
        gridSwp[id] = 1;
    else
        gridSwp[id] = current;
}
