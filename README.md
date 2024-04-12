# Calculating the Moore–Penrose Generalized Inverse on Massively Parallel Systems

> [**Calculating the Moore–Penrose Generalized Inverse on Massively Parallel Systems**](https://www.mdpi.com/1999-4893/15/10/348)
> 
> Vukašin Stanojević,Lev Kazakovtsev, Predrag S. Stanimirović, Natalya Rezova, Guzel Shkaberina

## Abstract
In this work, we consider the problem of calculating the generalized Moore–Penrose inverse, which is essential in many applications of graph theory. We propose an algorithm for the massively parallel systems based on the recursive algorithm for the generalized Moore–Penrose inverse, the generalized Cholesky factorization, and Strassen’s matrix inversion algorithm. Computational experiments with our new algorithm based on a parallel computing architecture known as the Compute Unified Device Architecture (CUDA) on a graphic processing unit (GPU) show the significant advantages of using GPU for large matrices (with millions of elements) in comparison with the CPU implementation from the OpenCV library. 
<p align="center"><img src="assets/overview.png" width="600"/></p>

# Citation

If you find our work useful, please cite our paper: 
```
@Article{a15100348,
  AUTHOR = {Stanojević, Vukašin and Kazakovtsev, Lev and Stanimirović, Predrag S. and Rezova, Natalya and Shkaberina, Guzel},
  TITLE = {Calculating the Moore–Penrose Generalized Inverse on Massively Parallel Systems},
  JOURNAL = {Algorithms},
  VOLUME = {15},
  YEAR = {2022},
  NUMBER = {10},
  ARTICLE-NUMBER = {348},
  URL = {https://www.mdpi.com/1999-4893/15/10/348},
  ISSN = {1999-4893},
  DOI = {10.3390/a15100348}
}

```
