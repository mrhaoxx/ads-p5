import sys
import random

def generate_knapsack_input(N, Capacity, max_weight_percentage, max_value=100):
    # Print the first line with the number of items and capacity
    print(f"{N} {Capacity}")

    # Calculate the maximum weight based on the percentage of Capacity
    max_weight = int(Capacity * max_weight_percentage / 100)

    # Generate N items with random weights and values
    for _ in range(N):
        weight = random.randint(1, max_weight)  # Ensure weights are within the calculated max
        value = random.randint(1, max_value)  # Values between 1 and max_value
        print(f"{weight} {value}")

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python knapsack_input_generator.py <N> <Capacity> <MaxWeightPercentage> [<MaxValue>]")
        sys.exit(1)

    try:
        N = int(sys.argv[1])
        Capacity = int(sys.argv[2])
        max_weight_percentage = float(sys.argv[3])
        max_value = int(sys.argv[4]) if len(sys.argv) == 5 else 100
        if N <= 0 or Capacity <= 0 or not (0 < max_weight_percentage <= 100) or max_value <= 0:
            raise ValueError("N and Capacity must be positive integers, MaxWeightPercentage must be between 0 and 100, and MaxValue must be positive.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    generate_knapsack_input(N, Capacity, max_weight_percentage, max_value)

if __name__ == "__main__":
    main()