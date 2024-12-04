#include "Knapsack.hpp"
#include <iostream>
#include <chrono>
#include <fstream>

template <typename Solver>
std::pair<result_t, long long> test(int n, int W, const std::vector<weight_t>& w, const std::vector<value_t>& v) {
    auto start = std::chrono::high_resolution_clock::now();

    result_t result = Solver::knapsack(n, W, w, v);

    auto end = std::chrono::high_resolution_clock::now();

    return {result, std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);

    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << argv[1] << std::endl;
        return 1;
    }

    int n, W;
    file >> n >> W;

    std::vector<weight_t> w(n);
    std::vector<value_t> v(n);

    for (int i = 0; i < n; i++) {
        file >> w[i] >> v[i];
    }

    file.close();

    // sort with best biggest value/weight ratio

    std::chrono::high_resolution_clock::time_point start_sort = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<double, int>> ratios(n);

    for (int i = 0; i < n; i++) {
        ratios[i] = {v[i] / (double)w[i], i};
    }

    std::sort(ratios.begin(), ratios.end(), std::greater<>());

    std::vector<weight_t> w_sorted(n);
    std::vector<value_t> v_sorted(n);

    for (int i = 0; i < n; i++) {
        w_sorted[i] = w[ratios[i].second];
        v_sorted[i] = v[ratios[i].second];
    }
    
    std::chrono::high_resolution_clock::time_point end_sort = std::chrono::high_resolution_clock::now();

    std::cout << "Sorting time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_sort - start_sort).count() << "us" << std::endl;

    // function ptr for test
    using test_fn =  std::pair<result_t, long long>(*)(int, int, const std::vector<weight_t>&, const std::vector<value_t>&);
    using test_s = std::pair<std::string, test_fn>;

    std::vector<test_s> tests = {
        {"Baseline", test<Baseline>},
        {"WithLists", test<WithLists>},
        {"WithBounds", test<WithBounds>},
    };

    result_t t = -1;

    for (const auto& [name, fn] : tests) {
        auto [result, time] = fn(n, W, w_sorted, v_sorted);
        if (t == -1) {
            t = result;
        } else {
            if (t != result) {
                std::cerr << "Error: " << name << " result is different from the baseline" << std::endl;
            }
        }
        std::cout << name << " result: " << result << " time: " << time << "us" << std::endl;
    }    

    return 0;
}