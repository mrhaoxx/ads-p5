#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <numeric>

using weight_t = int;
using result_t = int;
using value_t = int;
using state_t = std::pair<weight_t, value_t>;

class Baseline {
public:
     static result_t knapsack(int n, int W, const std::vector<weight_t>& w, const std::vector<value_t>& v) {
        std::vector<std::vector<result_t>> dp(n + 1, std::vector<result_t>(W + 1));
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= W; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 0;
                } else if (w[i - 1] <= j) {
                    dp[i][j] = std::max(v[i - 1] + dp[i - 1][j - w[i - 1]], dp[i - 1][j]);
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n][W];
    }
};

class WithLists {
public:

    static result_t knapsack(int n, int W, const std::vector<weight_t>& w, const std::vector<value_t>& p) {
        std::vector<state_t> L = { {0, 0} };

        for (int j = 0; j < n; j++) {
            std::vector<state_t> Lp;
            for (const auto& [weight, profit] : L) {
                if (weight + w[j] <= W) {
                    Lp.emplace_back(weight + w[j], profit + p[j]);
                }
            }

            L = mergelists(L, Lp);
        }

        result_t maxProfit = 0;
        for (const auto& [weight, profit] : L) {
            maxProfit = std::max(maxProfit, profit);
        }

        return maxProfit;
    }

    static std::vector<state_t> mergelists(const std::vector<state_t>& L1, const std::vector<state_t>& L2) {
        std::vector<state_t> result;
        size_t i = 0, j = 0;

        state_t largest = {-1, -1};
        size_t index = 0;

        while (i < L1.size() || j < L2.size()) {
            state_t next;

            if (j >= L2.size() || (i < L1.size() && L1[i].first <= L2[j].first)) {
                next = L1[i++];
            } else {
                next = L2[j++];
            }

            if (result.empty()){
                result.push_back(next);
                largest = next;
            } else {
                if (next.second > largest.second){
                    if (next.first > largest.first){
                        result.push_back(next);
                        largest = next;
                        index = result.size() - 1;
                    } else if (next.first == largest.first){
                        result[index] = next;
                        largest = next;
                    } else{
                        std::terminate();
                    }
                }
            }
        }

        return result;
    }
};


class WithBounds {
public:
    static result_t knapsack(int n, int W, const std::vector<weight_t>& w, const std::vector<value_t>& p) {
        std::vector<state_t> L = { {0, 0} }; 
        int z = 0;

        for (int j = 0; j < n; j++) {
            std::vector<state_t> Lp;
            for (const auto& [weight, profit] : L) {
                if (weight + w[j] <= W) {
                    Lp.emplace_back(weight + w[j], profit + p[j]);
                }
            }

            L = mergelists_withbounds(L, Lp, j, W, w, p, z);
        }

        return z;
    }

private:
    static std::vector<state_t> mergelists_withbounds(const std::vector<state_t>& L1, const std::vector<state_t>& L2, int j, int W, const std::vector<int>& w, const std::vector<int>& p, int& z) {
        std::vector<state_t> result;

        state_t largest = {-1, -1};
        size_t index = 0;
        
        size_t i = 0, k = 0;

        while (i < L1.size() || k < L2.size()) {
            state_t next;

            if (k >= L2.size() || (i < L1.size() && L1[i].first <= L2[k].first)) {
                next = L1[i++];
            } else {
                next = L2[k++];
            }

            result_t upperBound = computeUpperBound(next.first, next.second, j, W, w, p);

            if (upperBound > z) {
                if (result.empty()){
                    result.push_back(next);
                    largest = next;
                } else {
                    if (next.second > largest.second){
                        if (next.first > largest.first){
                            result.push_back(next);
                            largest = next;
                            index = result.size() - 1;

                            if (next.second > z) {
                                z = next.second;
                            }

                        } else if (next.first == largest.first) {
                            result[index] = next;
                            largest = next;
                            
                            if (next.second > z) {
                                z = next.second;
                            }
                        } 
                    }
                }
            }
        }

        return result;
    }


    // this methods requires the items to be sorted by value/weight ratio
    static result_t computeUpperBound(weight_t weight, value_t profit, int j, int W, const std::vector<weight_t>& w, const std::vector<value_t>& p) {
        weight_t remainingCapacity = W - weight;
        value_t upperBound = profit;

        for (int i = j + 1; i < w.size(); i++) {
            if (w[i] <= remainingCapacity) {
                upperBound += p[i];
                remainingCapacity -= w[i];
            } else {
                upperBound += p[i] * remainingCapacity / w[i];
                break;
            }
        }

        return upperBound;
    }
};

template <int Numerator, int Denominator>
class FPTAS {
public:
    static result_t knapsack(int n, int W, const std::vector<weight_t>& w, const std::vector<value_t>& v) {
        // Step 1: Calculate epsilon as a rational number
        constexpr double epsilon = static_cast<double>(Numerator) / Denominator;

        value_t max_profit = *std::max_element(v.begin(), v.end());
        double scale_factor = std::max(1.0, static_cast<double>(epsilon * (double)max_profit / n));

        std::vector<value_t> scaled_profits(n);
        for (int i = 0; i < n; ++i) {
            scaled_profits[i] = v[i] / scale_factor;
        }

        // Step 3: Compute upper bound for profits
        value_t UB = std::accumulate(scaled_profits.begin(), scaled_profits.end(), 0);

        // Step 4: Initialize DP table
        std::vector<weight_t> dp(UB + 1, std::numeric_limits<weight_t>::max());
        dp[0] = 0;

        // Step 5: Dynamic programming
        for (int i = 0; i < n; ++i) {
            for (int p = UB; p >= scaled_profits[i]; --p) {
                if (dp[p - scaled_profits[i]] != std::numeric_limits<weight_t>::max()) {
                    dp[p] = std::min(dp[p], dp[p - scaled_profits[i]] + w[i]);
                }
            }
        }

        // Step 6: Find maximum profit within weight limit
        value_t max_profit_within_limit = 0;
        for (value_t p = 0; p <= UB; ++p) {
            if (dp[p] <= W) {
                max_profit_within_limit = p;
            }
        }

        return max_profit_within_limit * scale_factor;
    }
};

class KASI_value
{
public:
    static result_t knapsack(int n, int W, const std::vector<weight_t> &w, const std::vector<value_t> &p)
    {
        value_t s = 0;
        for (int i = 0; i < n; i++)
        {
            s = std::max(s, w[i]);
        }
        std::cerr << "s: " << s << std::endl;

        const auto [_p, delta, pos] = get_max_prefix_sum(0, n, W, w, p);
        const auto max_k = 2 * s * s + delta;

        std::cerr << "p: " << _p << " delta: " << delta << " pos: " << pos << " max_k: " << max_k << std::endl;

        std::vector<int> pos_t(max_k + 1);
        std::vector<int> neg_t(max_k + 1);

        for (int k = 0; k <= 2 * s * s + delta; k++)
        {
            // get max sum v_i for i in [pos, n) with sum weight <= k
            std::vector<int> pos_xs_w, pos_xs_p;
            for (int i = pos; i < n; i++)
            {
                pos_xs_w.push_back(w[i]);
                pos_xs_p.push_back(p[i]);
            }

            pos_t[k] = Baseline::knapsack(pos_xs_w.size(), k, pos_xs_w, pos_xs_p);

            std::cout << " " <<  pos_t[k] << " " << solve_vx_eq(pos_xs_p, pos_xs_w, k, s) << "\n";

            // std::cerr << "k: " << k << " pos_t: ";
            // for (auto& x: pos_t[k]){
            //     std::cerr << x << " ";
            // }
            // std::cerr << "with sum ";

            // int sum = 0;
            // for (auto& x: pos_t[k]){
            //     sum += w[x];
            // }

            // std::cerr << sum << std::endl;

            // get max sum - v_i for i in [0, pos) with sum weight == k

            std::vector<int> neg_xs_w, neg_xs_p;
            for (int i = 0; i < pos; i++)
            {
                neg_xs_w.push_back(w[i]);
                neg_xs_p.push_back(-p[i]);
            }

            neg_t[k] = solve_vx_eq(neg_xs_p, neg_xs_w, k, s);
            
            std::cerr << "\r" << "k: " << k;
        }

        std::cerr << std::endl;

        result_t max_profit = 0;

        for (int k = 0; k <= 2 * s * s; k++)
        {
            max_profit = std::max(_p + neg_t[k] + pos_t[k + delta], max_profit);
        }

        return max_profit;
    }

private:
    static std::tuple<result_t, weight_t, size_t> get_max_prefix_sum(int j, int n, int W, const std::vector<weight_t> &w, const std::vector<value_t> &p)
    {
        result_t prefix_sum = 0;
        weight_t remaining_capacity = W;
        size_t pos = 0;

        for (int i = j; i < n; i++)
        {
            if (remaining_capacity - w[i] >= 0)
            {
                prefix_sum += p[i];
                remaining_capacity -= w[i];
            }
            else
            {
                pos = i;
                break;
            }
        }
        return {prefix_sum, remaining_capacity, pos};
    }

    static int maxTotalValueWithItems(const std::vector<int> &values, const std::vector<int> &weights, int targetWeight)
    {
        int n = values.size();
        std::vector<int> dp(targetWeight + 1, std::numeric_limits<int>::min());
        std::vector<int> prev(targetWeight + 1, -1);
        std::vector<int> itemIndex(targetWeight + 1, -1);
        dp[0] = 0; // 总重量为0时，总价值为0

        for (int i = 0; i < n; ++i)
        {
            // 从大到小遍历，避免重复使用物品
            for (int w = targetWeight; w >= weights[i]; --w)
            {
                if (dp[w - weights[i]] != std::numeric_limits<int>::min())
                {
                    int newValue = dp[w - weights[i]] + values[i];
                    if (newValue > dp[w])
                    {
                        dp[w] = newValue;
                        prev[w] = w - weights[i]; // 记录上一个重量
                        itemIndex[w] = i;         // 记录选择的物品索引
                    }
                }
            }
        }

        std::cout << "direct: " <<dp[targetWeight] << " ";
        return dp[targetWeight];

    }

    static int solve_vx_le(const std::vector<int> &values, const std::vector<int> &weights, int targetWeight, int s)
    {
        // Initialize partitions for weights from 1 to s
        std::vector<std::vector<int>> partitions(s + 1);

        for (int i = 0; i < values.size(); i++)
        {
            partitions[weights[i]].push_back(values[i]);
        }

        // Initialize omega_eq and omega_le with minimum integer values
        std::vector<std::vector<int>> omega_eq(s + 1, std::vector<int>(targetWeight + 1, std::numeric_limits<int>::min()));
        std::vector<std::vector<int>> omega_le(s + 1, std::vector<int>(targetWeight + 1, std::numeric_limits<int>::min()));

        // Base case initialization for h = 0
        omega_eq[0][0] = 0;
        omega_le[0][0] = 0;

        // Compute omega_eq for each weight h
        for (int h = 0; h <= s; h++)
        {
            int cur_value = 0;
            int max_items = partitions[h].size();
            for (int i = 0; i <= max_items; i++)
            {
                int weight = i * h;
                if (weight > targetWeight)
                    break;

                omega_eq[h][weight] = cur_value;

                if (i == max_items)
                    break;

                cur_value += partitions[h][i];
            }
        }

        // Compute omega_le using dynamic programming
        for (int h = 1; h <= s; h++)
        {
            for (int i = 0; i <= targetWeight; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    if (omega_eq[h-1][j] != std::numeric_limits<int>::min() && omega_le[h - 1][i - j] != std::numeric_limits<int>::min())
                    {
                        omega_le[h][i] = std::max(omega_eq[h-1][j] + omega_le[h - 1][i - j], omega_le[h][i]);
                    }
                }
            }
        }


        int max = 0;
        for(int i = 0; i <= targetWeight;i++) 
            max = std::max(max, omega_le[s][i]);
        // std::cout << "conv: " << omega_le[s][targetWeight] << std::endl;

        // Return the result (modify as per your requirements)
        return max;
    }

    using element_t = std::pair<int,int>;

    // static std::vector<int> solve_vx_eq(const std::vector<int> &values, const std::vector<int> &weights, int targetWeight, int s)
    // {
    //     std::vector<std::vector<element_t>> partitions(s+1);
    //     for (int i = 0; i < values.size(); i++)
    //     {
    //         partitions[weights[i]].push_back({i, values[i]});
    //     }

    //     std::vector<std::vector<int>> omega_eq(s+1, std::vector<int>(targetWeight+1, std::numeric_limits<int>::min()));
    //     std::vector<std::vector<int>> omega_le(s+1, std::vector<int>(targetWeight+1, std::numeric_limits<int>::min()));

    //     for (int h = 0; h <= s;h++){
    //         int cur_value = 0;
    //         for(int i = 0; i <= partitions[h].size(); i++){
    //             if (i * h > targetWeight)
    //                 break;

    //             omega_eq[h][i*h] = cur_value;

    //             if (i == partitions[h].size())
    //                 break;

    //             cur_value += partitions[h][i].second;
    //         }
    //     }

    //     for(int h = 1; h <= s; h++){
    //         for(int i = 0; i < targetWeight+1; i++){
    //            for(int j=0; j <= i;j++){
    //                 omega_le[h][i] = std::max(omega_le[h][i], omega_eq[h-1][j] + omega_le[h-1][i-j]);
    //            } 
    //         }
    //     }

    //     std::cout << "conv: " << omega_le[s][targetWeight] << std::endl;

    // }
    // Define element_t as a pair of integers
    static int solve_vx_eq(const std::vector<int> &values, const std::vector<int> &weights, int targetWeight, int s)
    {
        // Initialize partitions for weights from 1 to s
        std::vector<std::vector<int>> partitions(s + 1);

        for (int i = values.size() - 1; i >=0; i--)
        {
            partitions[weights[i]].push_back(values[i]);
        }

        // Initialize omega_eq and omega_le with minimum integer values
        std::vector<std::vector<int>> omega_eq(s + 1, std::vector<int>(targetWeight + 1, std::numeric_limits<int>::min()));
        std::vector<std::vector<int>> omega_le(s + 1, std::vector<int>(targetWeight + 1, std::numeric_limits<int>::min()));

        // Base case initialization for h = 0
        omega_eq[0][0] = 0;
        omega_le[0][0] = 0;

        // Compute omega_eq for each weight h
        for (int h = 0; h <= s; h++)
        {
            int cur_value = 0;
            int max_items = partitions[h].size();
            for (int i = 0; i <= max_items; i++)
            {
                int weight = i * h;
                if (weight > targetWeight)
                    break;

                omega_eq[h][weight] = cur_value;

                if (i == max_items)
                    break;

                cur_value += partitions[h][i];
            }
        }

        // Compute omega_le using dynamic programming
        for (int h = 1; h <= s; h++)
        {
            for (int i = 0; i <= targetWeight; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    if (omega_eq[h-1][j] != std::numeric_limits<int>::min() && omega_le[h - 1][i - j] != std::numeric_limits<int>::min())
                    {
                        omega_le[h][i] = std::max(omega_eq[h-1][j] + omega_le[h - 1][i - j], omega_le[h][i]);
                    }
                }
            }
        }


        // std::cout << "conv: " << omega_le[s][targetWeight] << std::endl;

        // Return the result (modify as per your requirements)
        return omega_le[s][targetWeight];
    }
};
