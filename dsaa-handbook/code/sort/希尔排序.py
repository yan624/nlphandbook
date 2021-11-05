import random
import time

from utils import random_list, is_sort


def sort(arr):
    n = len(arr)
    gap = n // 2
    while gap != 0:
        for i in range(gap, gap * 2):
            insertion_sort(i, gap, arr)
        gap //= 2


# def insertion_sort(start, gap, arr):
#     """
#     错误的做法
#     """
#     for i in range(start, len(arr), gap):
#         j, tmp = i, arr[i]
#         while j > 0:
#             # 如果不符合条件，就不让进循环，也就是不让 j-gap。
#             if tmp < arr[j - gap]:
#                 arr[j] = arr[j - gap]
#             j -= gap
#             # 如果非要用这种格式，应该这么写
#             # if tmp < arr[j - gap]:
#             #     arr[j] = arr[j - gap]
#             # else:
#             #     j -= gap
#         arr[j] = tmp


def insertion_sort(start, gap, arr):
    """
    正确的做法
    """
    for i in range(start, len(arr), gap):
        j, tmp = i, arr[i]
        while j > 0 and tmp < arr[j - gap]:
            arr[j] = arr[j - gap]
            j -= gap
        arr[j] = tmp


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