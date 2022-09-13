//
// Created by vukasin on 7/9/21.
//

#ifndef MOOREPENROSE_FUNCTORS_H
#define MOOREPENROSE_FUNCTORS_H

template<typename T>
struct Add {
    __host__ __device__ static T f(const T& a, const T& b) {return a + b;}
};

template<typename T>
struct Subtract {
    __host__ __device__ static T f(const T& a, const T& b) {return a - b;}
};

#endif //MOOREPENROSE_FUNCTORS_H
