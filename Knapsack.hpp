#include <vector>
#include <algorithm>
#include <iostream>

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

            int upperBound = computeUpperBound(next.first, next.second, j, W, w, p);

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
    static int computeUpperBound(int weight, int profit, int j, int W, const std::vector<int>& w, const std::vector<int>& p) {
        int remainingCapacity = W - weight;
        int upperBound = profit;

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


class VMimproved {
public:
};