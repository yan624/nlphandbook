import random
import time

from utils import random_list


def sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(1, n - i):
            if arr[j - 1] > arr[j]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]


# random.seed(123)
test_arr = random_list(10000, 0, 100000)
# test_arr = [i for i in range(10000)]
print(test_arr)
start = time.time()
sort(test_arr)
end = time.time()
print(test_arr)
print('%.2fs' % (end - start))