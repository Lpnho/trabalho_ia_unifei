#include <iostream>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <sstream>
#include <random>
#include <omp.h>
#include <iomanip>
#include <array>

template <std::size_t N>
class MemoryData
{
    std::array<double, N> _memory;

public:
    MemoryData = default;

    MemoryData(std::initializer_list<double> data)
    {
        std::copy_n(data.begin(), std::min(data.size(), N), _memory.begin());
    }

    MemoryData clone()
    {
        MemoryData result;
        std::copy_n(_memory.begin(), N, result._memory.begin());
        return result;
    }
    MemoryData& copy(const MemoryData &other)
    {
        std::copy_n(other._memory.begin(), N, _memory.begin());
        return *this;
    }
};

int main()
{
    return EXIT_SUCCESS;
}
