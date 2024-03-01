import numpy as np
import timeit

def squares_list(N):
    return [i**2 for i in range(N)]

def squares_loop(N):
    squares = np.zeros(N)
    for i in range(N):
        squares[i] = i**2
    return squares

def squares_vectorized(N):
    return np.arange(N)**2

N = 100000

start = timeit.default_timer()
squares_list(N)
end = timeit.default_timer()
print(f"Time for list: {end - start} seconds")

start = timeit.default_timer()
squares_loop(N)
end = timeit.default_timer()
print(f"Time for loop: {end - start} seconds")

start = timeit.default_timer()
squares_vectorized(N)
end = timeit.default_timer()
print(f"Time for vectorized: {end - start} seconds")