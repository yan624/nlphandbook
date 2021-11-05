import random
import time

from utils import random_list, is_sort


def sort(arr):
    if len(arr) <= 10:
        arr.sort()
        return
    pivot = (arr[0] + arr[len(arr) // 2] + arr[-1]) // 3
    s1, s2 = [], []
    for item in arr:
        if item <= pivot:
            s1.append(item)
        else:
            s2.append(item)
    sort(s1)
    sort(s2)
    arr.clear()
    arr.extend(s1)
    arr.extend(s2)


# random.seed(123)
test_arr = random_list(10000, 0, 100000)
# test_arr = [i for i in range(10000)]
print(test_arr)
start = time.time()
sort(test_arr)
end = time.time()
print(test_arr)
print(is_sort(test_arr))
print('%.2fs' % (end - start))