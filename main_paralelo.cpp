#include <iostream>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <sstream>
#include <random>
#include <omp.h>
#include <iomanip>
#include <chrono>

template <std::size_t N>
inline double norm(double (&elemento)[N])
{
    double acc{0};
#pragma omp simd reduction(+ : acc)
    for (std::size_t i = 0; i < N; ++i)
    {
        acc += elemento[i] * elemento[i];
    }
    return std::sqrt(acc);
}

template <std::size_t N>
inline void sub(double (&minuendo)[N], double (&subtraendo)[N], double (&resto)[N])
{
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i)
    {
        resto[i] = minuendo[i] - subtraendo[i];
    }
}

template <std::size_t N>
inline void add(double (&a)[N], double (&b)[N], double (&result)[N])
{
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = a[i] + b[i];
    }
}

template <std::size_t N>
inline double mult(double (&a)[N], double (&b)[N])
{
    double acc{0};
#pragma omp simd reduction(+ : acc)
    for (std::size_t i = 0; i < N; ++i)
    {
        acc += a[i] * b[i];
    }
    return acc;
}

template <std::size_t N>
inline void mult(double (&a)[N], double b, double (&result)[N])
{
#pragma omp simd
    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = a[i] * b;
    }
}

template <std::size_t N>
inline void orthogonalProjection(double (&a)[N], double (&b)[N], double (&result)[N])
{
    mult(b, mult(a, b), result);
}

constexpr double divideByZeroProtection(double fator)
{
    return (fator > 1.0e-12 ? fator : 1.0e-6);
}

template <std::size_t N>
struct ParentCentricCrossover
{
    template <std::size_t u>
    static void PCX(double (&dads)[u][N], double (&filho)[N], double sigmaKsi, double sigmaEta)
    {
        thread_local static std::random_device device{};
        thread_local static std::mt19937 gerador{device()};
        thread_local static std::uniform_int_distribution<std::size_t> rand(0, u - 1);
        thread_local static std::normal_distribution<double> randDouble(0.0, 1.0);

        double g[N]{};
        double d[N]{};

        std::size_t indexDad = rand(gerador);
        double *p = dads[indexDad];

        for (std::size_t iDad = 0; iDad < u; ++iDad)
        {
            for (std::size_t iN = 0; iN < N; iN++)
            {
                g[iN] += dads[iDad][iN] / u;
            }
        }

        for (std::size_t iN = 0; iN < N; iN++)
        {
            d[iN] = p[iN] - g[iN];
        }

        double normd = norm(d);
        double normdQuad = divideByZeroProtection(normd * normd);

        double w[u - 1][N]{};
        double v[N]{};
        double di[N]{};
        double Dmed = 0.0;
        std::size_t k = 0;

        for (std::size_t iDad = 0; iDad < u; ++iDad)
        {
            if (iDad != indexDad)
            {
                sub(dads[iDad], g, v);
                mult(d, mult(v, d) / normdQuad, di);
                sub(v, di, w[k]);
                Dmed += norm(w[k]) / (u - 1);
                ++k;
            }
        }
        double e[u - 1][N]{};
        double sumAcc[N]{};
        double orthogonalProjectionResult[N]{};

        mult(w[0], (1.0 / (divideByZeroProtection(norm(w[0])))), e[0]);

        for (std::size_t iW = 1; iW < u - 1; ++iW)
        {
            orthogonalProjection(w[iW], e[0], sumAcc);

            for (std::size_t iN = 1; iN < iW; ++iN)
            {
                orthogonalProjection(w[iW], e[iN], orthogonalProjectionResult);
                add(orthogonalProjectionResult, sumAcc, sumAcc);
            }
            sub(w[iW], sumAcc, e[iW]);

            mult(e[iW], (1.0 / (divideByZeroProtection(norm(e[iW])))), e[iW]);
        }

        double sumT[N]{};
        double omegaKsi = randDouble(gerador) * sigmaKsi;
        double K = sigmaEta * Dmed;
        for (std::size_t iW = 1; iW < u - 1; ++iW)
        {
            double omegaEta = randDouble(gerador);
            for (std::size_t iN = 0; iN < N; iN++)
            {
                sumT[iN] += e[iW][iN] * omegaEta * K;
            }
        }

        for (std::size_t iN = 0; iN < N; iN++)
        {
            filho[iN] = p[iN] + omegaKsi * d[iN] + sumT[iN];
        }
    }
};

template <std::size_t N,
          std::size_t POPULATION_SIZE>
class GeneticAlgorithm
{
    const double MAX = 5.12;
    const double CROSSOVER_RATE = .4;
    static constexpr std::size_t TOURNAMENT_N = 3;
    const double TOURNAMENT_K = 0.75;

    const double MUTATE_RATE = 0.1;
    const double MUTATE_N_RATE = 0.4;
    const double MUTATION_SIGMA = 1.0;

    static constexpr std::size_t CROSSOVER_U = 8;
    static constexpr double SIGMA_KSI = 0.1;
    static constexpr double SIGMA_ETA = 0.1;

    const std::size_t GENERATION_N = 200;

    const double A = 10;

    static constexpr std::size_t BUFFERS_COUNT = 2;
    double _population[BUFFERS_COUNT][POPULATION_SIZE][N];

public:
    GeneticAlgorithm &initPopulation(double (&population)[POPULATION_SIZE][N])
    {
        std::random_device _random_device;
        std::mt19937 _gerador{_random_device()};
        std::uniform_real_distribution<double> random(-MAX, MAX);
        for (std::size_t iPop = 0; iPop < POPULATION_SIZE; ++iPop)
        {
            for (std::size_t iN = 0; iN < N; ++iN)
            {
                population[iPop][iN] = random(_gerador);
            }
        }
        return *this;
    }

    GeneticAlgorithm &printPopulation(double (&population)[POPULATION_SIZE][N], std::size_t precision = 8)
    {
        std::stringstream buffer;
        buffer << std::setprecision(precision);
        for (std::size_t iPop = 0; iPop < POPULATION_SIZE; ++iPop)
        {
            for (std::size_t iN = 0; iN < N; ++iN)
            {
                buffer << population[iPop][iN] << ' ';
            }
            buffer << '\n';
        }
        std::cout << buffer.str();
        return *this;
    }

    bool comparePopulation(double (&populationA)[POPULATION_SIZE][N], double (&populationB)[POPULATION_SIZE][N])
    {
        for (std::size_t iPop = 0; iPop < POPULATION_SIZE; ++iPop)
        {
            for (std::size_t iN = 0; iN < N; ++iN)
            {
                if (populationA[iPop][iN] != populationB[iPop][iN])
                    return false;
            }
        }
        return true;
    }

    inline double applyFitness(double (&population)[POPULATION_SIZE][N], std::size_t individuo)
    {
        double sum = 0.0;
        for (std::size_t elemento = 0; elemento < N; ++elemento)
        {
            sum += population[individuo][elemento] * population[individuo][elemento] -
                   A * std::cos(population[individuo][elemento] * std::numbers::pi * 2);
        }
        return sum + A * N;
    }

    template <std::size_t NT>
    inline std::size_t tournamentSelection(double (&population)[POPULATION_SIZE][N])
    {
        thread_local static std::random_device _random_device;
        thread_local static std::mt19937 _gerador{_random_device()};
        thread_local static std::uniform_int_distribution<std::size_t> random(0, POPULATION_SIZE - 1);
        thread_local static std::uniform_real_distribution<double> randomDouble(0.0, 1.0);
        thread_local static std::size_t elements[NT]{};
        std::size_t maior = 0;
        std::size_t menor = 0;

        elements[0] = random(_gerador);
        double maiorFit = applyFitness(population, elements[0]);
        double menorFit = maiorFit;
        double fit = 0;

        for (std::size_t index = 1; index < NT; ++index)
        {
            elements[index] = random(_gerador);
            fit = applyFitness(population, elements[index]);
            if (fit > maiorFit)
            {
                maior = index;
                maiorFit = fit;
            }
            else if (fit < menorFit)
            {
                menor = index;
                menorFit = fit;
            }
        }
        // return ((randomDouble(_gerador) < TOURNAMENT_K) ? maior : menor);
        return ((randomDouble(_gerador) < TOURNAMENT_K) ? menor : maior);
    }

    inline void gaussianMutation(double (&input)[POPULATION_SIZE][N],
                                 double (&output)[POPULATION_SIZE][N], std::size_t element)
    {
        thread_local static std::random_device _random_device;
        thread_local static std::mt19937 _gerador{_random_device()};
        thread_local static std::uniform_real_distribution<double> random(0.0, 1.0);
        thread_local static std::normal_distribution<double> normDist(0.0, MUTATION_SIGMA);

        for (std::size_t index = 0; index < N; ++index)
        {
            output[element][index] = input[element][index];

            if (random(_gerador) < MUTATE_N_RATE)
            {
                output[element][index] += normDist(_gerador);
            }
        }
    }

    inline void run(std::size_t precision = 8)
    {
        std::stringstream logBuffer;
        logBuffer << std::fixed << std::setprecision(precision);

        thread_local static std::random_device randomDevice;
        thread_local static std::mt19937 gerador{randomDevice()};
        thread_local static std::uniform_real_distribution<double> randomDouble(0.0, 1.0);
        thread_local static std::uniform_int_distribution<std::size_t> random(0, POPULATION_SIZE - 1);

        std::size_t currentPopulation = 0;
        std::size_t bufferPopulation = 1;

        initPopulation(_population[currentPopulation]);

        for (std::size_t geracoes = 0; geracoes < GENERATION_N; ++geracoes)
        {
            double currentBestFitness;
            std::size_t currentBestIndex = -1;
            bool bestInitialized = false;
#pragma omp parallel
            {
                double dads[CROSSOVER_U][N]{};

                double localCurrentBestFitness;
                std::size_t localCurrentBestIndex;

                double value;
                bool crossOverOrMutation;

                localCurrentBestFitness = applyFitness(_population[currentPopulation], 0);
                localCurrentBestIndex = 0;
#pragma omp for
                for (std::size_t individuo = 1; individuo < POPULATION_SIZE; ++individuo)
                {
                    crossOverOrMutation = false;

                    value = applyFitness(_population[currentPopulation], individuo);

                    if (value < localCurrentBestFitness)
                    {
                        localCurrentBestFitness = value;
                        localCurrentBestIndex = individuo;
                    }

                    if (randomDouble(gerador) < CROSSOVER_RATE)
                    {
                        crossOverOrMutation = true;
                        for (std::size_t dadIndex = 0; dadIndex < CROSSOVER_U; ++dadIndex)
                        {
                            std::size_t randDad = tournamentSelection<TOURNAMENT_N>(_population[currentPopulation]);
                            for (std::size_t iN = 0; iN < N; ++iN)
                            {
                                dads[dadIndex][iN] = _population[currentPopulation][randDad][iN];
                            }
                        }
                        ParentCentricCrossover<N>::PCX(dads, _population[bufferPopulation][individuo], SIGMA_KSI, SIGMA_ETA);
                    }
                    else if (randomDouble(gerador) < MUTATE_RATE)
                    {
                        crossOverOrMutation = true;

                        gaussianMutation(
                            _population[currentPopulation],
                            _population[bufferPopulation],
                            individuo);
                    }

                    for (std::size_t iN = 0; iN < N; ++iN)
                    {
                        if (!crossOverOrMutation)
                        {
                            _population[bufferPopulation][individuo][iN] =
                                _population[currentPopulation][individuo][iN];
                        }

                        value = _population[bufferPopulation][individuo][iN];
                        value = std::max(-MAX, std::min(MAX, value));
                        _population[bufferPopulation][individuo][iN] = value;
                    }
                }

#pragma omp critical
                {
                    if (!bestInitialized)
                    {
                        bestInitialized = true;
                        currentBestIndex = localCurrentBestIndex;
                        currentBestFitness = localCurrentBestFitness;
                    }
                    else if (currentBestFitness > localCurrentBestFitness)
                    {
                        currentBestIndex = localCurrentBestIndex;
                        currentBestFitness = localCurrentBestFitness;
                    }
                }
            }
            logBuffer << currentBestFitness << ", " << geracoes << ' ';

            for (std::size_t iN = 0; iN < N; ++iN)
            {
                _population[bufferPopulation][0][iN] = _population[currentPopulation][currentBestIndex][iN];
                // logBuffer << _population[bufferPopulation][0][iN] << ' ';
            }
            logBuffer << '\n';
            // logBuffer << comparePopulation(_population[bufferPopulation], _population[currentPopulation]) << '\n';
            bufferPopulation = currentPopulation;
            currentPopulation = (currentPopulation + 1) % BUFFERS_COUNT;
        }
        std::cout << logBuffer.str();
    }
};

int main()
{
    auto inicio = std::chrono::steady_clock::now();
    GeneticAlgorithm<8, 2000>().run(16);
    auto fim = std::chrono::steady_clock::now();
    auto tempo =
        std::chrono::duration_cast<std::chrono::milliseconds>(fim - inicio);

    std::cout << "Tempo: " << tempo.count() << " ms\n";

    // GeneticAlgorithm<2, 10>().initPopulation().printPopulation();

    // double input[3][2] = {
    //     {0, 0},
    //     {2, 0},
    //     {0, 2},
    // };
    // double output[2];
    // ParentCentricCrossover<2>()
    //     .PCX(input, output, 0.2, 0.2);
    return EXIT_SUCCESS;
}
