from numba import njit
import time

if __name__ == '__main__':
    def test():
        start = time.time()
        alpha = {i: [j for j in range(10000)] for i in range(10000)}
        total = 0
        for x in alpha.keys():
            for y in alpha[x]:
                total + y

        print(time.time() - start)

    test()

    def test2():
        start = time.time()
        alpha = {i: [j for j in range(10000)] for i in range(10000)}
        total = 0
        for x in alpha.keys():
            for y in alpha[x]:
                total + y

        print(time.time() - start)