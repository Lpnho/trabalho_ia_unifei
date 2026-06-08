#include <iostream>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <sstream>
#include <random>
#include <omp.h>

template <std::size_t N,
          std::size_t POPULATION_SIZE,
          std::size_t ENCODER_PRECISION>
class GeneticAlgorithm
{
    const std::uint32_t FACTOR_ASCII = 48;
    const double MAX = 5.12;
    const double MUTATE_RATE = 0.1;
    const double MUTATE_STRONG_RATE = 0.05;
    const double CROSSOVER_RATE = .4;

    char _population[N * ENCODER_PRECISION * POPULATION_SIZE];

public:
    inline void print()
    {
        std::stringstream buffer;

        for (std::size_t individuo = 0; individuo < POPULATION_SIZE; ++individuo)
        {
            for (std::size_t elemento = 0; elemento < N; ++elemento)
            {
                std::size_t inicio = (elemento * ENCODER_PRECISION) + individuo * N * ENCODER_PRECISION;
                for (std::size_t indice = inicio; indice < inicio + ENCODER_PRECISION; ++indice)
                {
                    buffer << _population[indice];
                }
                buffer << ' ';
            }
            buffer << '\n';
        }
        std::cout << buffer.str();
    }

    inline double decode(std::size_t individuo, std::size_t index)
    {
        std::size_t begin = (index * ENCODER_PRECISION) + individuo * N * ENCODER_PRECISION;
        std::size_t end = begin + ENCODER_PRECISION;
        std::size_t i = begin + ((_population[begin] == '-') ? 1 : 0);

        double result = ((double)(_population[i] - FACTOR_ASCII));
        double multi = 1.0 / 10.0;
        for (i += 2; i < end; ++i)
        {
            result += multi * ((double)(_population[i] - FACTOR_ASCII));
            multi /= 10;
        }
        return result;
    }

    inline const char *encode(std::size_t individuo, std::size_t index, double value)
    {
        static char printData[ENCODER_PRECISION + 1];
        printData[ENCODER_PRECISION] = '\0';

        std::size_t begin = (index * ENCODER_PRECISION) + individuo * N * ENCODER_PRECISION;
        std::size_t end = begin + ENCODER_PRECISION;

        std::size_t i = begin;
        if ((value < 0))
        {
            value *= -1;
            printData[0] = _population[begin] = '-';
            ++i;
        }
        double copyValue = value;
        int intValue = (int)value;

        printData[i - begin] = _population[i] = (char)(intValue + FACTOR_ASCII);
        copyValue = 10 * (copyValue - intValue);
        intValue = (int)copyValue;

        ++i;

        printData[i - begin] = _population[i] = '.';

        for (i += 1; i < end; ++i)
        {
            printData[i - begin] = _population[i] = (char)(intValue + FACTOR_ASCII);
            copyValue = 10 * (copyValue - intValue);
            intValue = (int)copyValue;
        }
        return printData;
    }

private:
    double _A;

public:
    GeneticAlgorithm(double A = 10) : _A{A}
    {
        iniciarPopulacao();
    }
    GeneticAlgorithm &iniciarPopulacao()
    {
        std::random_device _random_device;
        std::mt19937 _gerador{_random_device()};
        std::uniform_real_distribution<double> random(-MAX, MAX);
        for (std::size_t individuo = 0; individuo < POPULATION_SIZE; ++individuo)
        {
            for (std::size_t elemento = 0; elemento < N; ++elemento)
            {
                encode(individuo, elemento, random(_gerador));
            }
        }
        return *this;
    }

private:
    inline double applyFitness(std::size_t individuo)
    {
        double sum = 0.0;
        double el = 0;
        for (std::size_t elemento = 0; elemento < N; ++elemento)
        {
            el = decode(individuo, elemento);
            sum += el * el - _A * std::cos(el * std::numbers::pi * 2);
        }
        return sum;
    }
    // inline void crossover(std::size_t individuoA, std::size_t individuoB)
    // {
    //     static std::random_device _random_device;
    //     static std::mt19937 _gerador{_random_device()};
    //     static std::uniform_int_distribution<double> randomIndex(0, ENCODER_PRECISION);

    //     std::size_t begin = (0 * ENCODER_PRECISION) + individuo * N * ENCODER_PRECISION;
    //     std::size_t end = begin + ENCODER_PRECISION * N;

    //     std::size_t index1 = randomIndex(_gerador);
    //     std::size_t index2 = randomIndex(_gerador);
    //     if (_population[begin + index1] == '-' || _population[begin + index1] == '.' ||
    //         _population[begin + index2] == '-' || _population[begin + index2] == '.')
    //         return;

    //     char aux = _population[begin + index1];
    //     _population[begin + index1] = _population[begin + index2];
    //     _population[begin + index2] = aux;
    // }

    inline void mutate(std::size_t individuo)
    {
        static std::random_device _random_device{};
        static std::mt19937 _gerador{_random_device()};
        static std::uniform_int_distribution<double> randomIndex(0, ENCODER_PRECISION);

        std::size_t begin = (0 * ENCODER_PRECISION) + individuo * N * ENCODER_PRECISION;
        std::size_t end = begin + ENCODER_PRECISION * N;

        std::size_t index1 = randomIndex(_gerador);
        std::size_t index2 = randomIndex(_gerador);
        if (_population[begin + index1] == '-' || _population[begin + index1] == '.' ||
            _population[begin + index2] == '-' || _population[begin + index2] == '.')
            return;

        char aux = _population[begin + index1];
        _population[begin + index1] = _population[begin + index2];
        _population[begin + index2] = aux;
    }
};

// Fitness
// Crossover
// Mutation
// Selection
// Encode
// Decode

template <std::size_t NN>
inline double Rastrigin(double A, std::array<double, NN> vecParam)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < NN; ++i)
    {
        sum += vecParam[i] * vecParam[i] - A * std::cos(vecParam[i] * std::numbers::pi * 2);
    }
    return A * NN + sum;
}

int main()
{
    GeneticAlgorithm<4, 2, 10>().print();
    // auto data = DataStorage<2, 2, 4>();
    // std::printf("%s, ", data.encode(0, 0, 3.1415));
    // std::printf("%s, ", data.encode(0, 1, 3.20));

    // std::printf("%s, ", data.encode(1, 0, 3.1415));
    // std::printf("%s, ", data.encode(1, 1, 3.20));

    return EXIT_SUCCESS;
}