import random
import time

from utils import random_list


def sort(arr):
    n = len(arr)
    for i in range(n):
        k = i
        for j in range(i, n):
            k = j if arr[j] < arr[k] else k
        arr[i], arr[k] = arr[k], arr[i]


# random.seed(123)
test_arr = random_list(10000, 0, 100000)
# test_arr = [i for i in range(10000)]
print(test_arr)
start = time.time()
sort(test_arr)
end = time.time()
print(test_arr)
print('%.2fs' % (end - start))