# Genetic Algorithm with Parent-Centric Crossover (PCX)

## Overview

This project implements a real-valued Genetic Algorithm (GA) in modern C++ focused on solving continuous optimization problems. The algorithm was developed to minimize the Rastrigin function, a well-known multimodal benchmark widely used to evaluate evolutionary optimization techniques.

The implementation combines:

* Tournament Selection
* Parent-Centric Crossover (PCX)
* Gaussian Mutation
* Elitism
* OpenMP Parallelization
* SIMD Optimizations

The objective is to achieve efficient exploration and exploitation of the search space while maintaining high computational performance.

---

## Problem

The algorithm minimizes the Rastrigin function:

[
f(x) = An + \sum_{i=1}^{n} \left(x_i^2 - A\cos(2\pi x_i)\right)
]

where:

* (A = 10)
* (n) is the dimensionality of the problem

The global optimum occurs at:

[
x = (0,0,\ldots,0)
]

with:

[
f(x)=0
]

---

## Main Features

### Parent-Centric Crossover (PCX)

The crossover operator generates offspring around a selected parent while considering the geometric distribution of multiple parents.

Advantages:

* Preserves useful genetic information.
* Improves local search capability.
* Suitable for continuous optimization problems.

### Tournament Selection

Parents are selected using tournament selection with configurable tournament size and selection pressure.

### Gaussian Mutation

Mutation introduces diversity by adding Gaussian noise to genes according to predefined mutation probabilities.

### Elitism

The best individual of each generation is preserved to guarantee that solution quality never degrades.

### OpenMP Parallelization

Population evolution is parallelized using OpenMP:

* Fitness evaluation
* Selection
* Crossover
* Mutation

This significantly reduces execution time for large populations.

### SIMD Optimization

Vector operations such as:

* Dot products
* Norm calculations
* Vector additions
* Vector subtractions

are accelerated using OpenMP SIMD directives.

---

## Algorithm Parameters

Default configuration:

| Parameter              | Value |
| ---------------------- | ----- |
| Population Size        | 10000 |
| Generations            | 1200  |
| Dimensions             | 10    |
| Crossover Rate         | 0.40  |
| Mutation Rate          | 0.10  |
| Gene Mutation Rate     | 0.40  |
| Mutation Sigma         | 1.0   |
| Tournament Size        | 3     |
| Tournament Probability | 0.75  |
| PCX Parents            | 8     |
| Sigma Ksi              | 0.1   |
| Sigma Eta              | 0.1   |

---

## Compilation

### GCC

```bash
g++ -O3 -march=native -fopenmp main.cpp -o ga
```

### Clang

```bash
clang++ -O3 -march=native -fopenmp main.cpp -o ga
```

---

## Execution

```bash
./ga
```

Example output:

```text
0.000000123
Tempo: 542 ms
```

---

## Project Structure

```text
├── main.cpp
├── README.md
└── results/
```

Main components:

* Vector algebra utilities
* PCX crossover implementation
* Genetic algorithm core
* Fitness evaluation
* Parallel execution support

---

## Performance Considerations

The implementation was designed with performance as a primary goal:

* Static memory allocation
* Minimal heap usage
* Thread-local random number generators
* OpenMP parallel regions
* SIMD vectorization
* Cached fitness values

These optimizations allow the algorithm to handle large populations efficiently.

---

## References

Deb, K., Anand, A., Joshi, D. (2002).

"A Computationally Efficient Evolutionary Algorithm for Real-Parameter Optimization".

IEEE Transactions on Evolutionary Computation.

Rastrigin, L. A. (1974).

"Systems of Extremal Control".
