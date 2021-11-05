import random
import time

from utils import random_list


"""
错误代码。
内部循环不要用 for 循环。
python 的 for 循环没办法到 0 自动退出。
"""
# def insert_sort(arr):
#     n = len(arr)
#     for i in range(1, n):
#         temp, j = arr[i], i
#         for j in range(i, 0, -1):
#             if temp > arr[j - 1]:
#                 break
#             arr[j] = arr[j - 1]
#         arr[j] = temp


def sort(arr):
    n = len(arr)
    for i in range(1, n):
        temp, j = arr[i], i
        while j > 0 and temp < arr[j - 1]:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = temp


# random.seed(123)
test_arr = random_list(10000, 0, 100000)
# test_arr = [i for i in range(10000)]
print(test_arr)
start = time.time()
sort(test_arr)
end = time.time()
print(test_arr)
print('%.2fs' % (end - start))

