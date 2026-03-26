struct Bodies {
    // Device arrays (each N elements)
    float* pos_x;      // positions
    float* pos_y;
    float* vel_x;      // velocities
    float* vel_y;
    float* acc_x;      // accelerations
    float* acc_y;
    float* mass;        // mass
    float* radius;      // collision radius = cbrt(mass)
    int    count;       // number of active bodies
    int    capacity;    // allocated capacity
};