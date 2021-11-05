import random
import time

from utils import random_list, is_sort


# def sort(i, j, arr):
#     i_, j_ = i + 1, j
#     if j_ - i_ <= 0:
#         return
#
#     pivot = arr[i]
#     while i_ != j_:
#         while arr[j_] >= pivot and i_ != j_:
#             j_ -= 1
#         while arr[i_] < pivot and i_ != j_:
#             i_ += 1
#         arr[i_], arr[j_] = arr[j_], arr[i_]
#     arr[i], arr[i_] = arr[i_], arr[i]
#     sort(i, i_ - 1, arr)
#     sort(i_ + 1, j, arr)


def sort(i, j, arr):
    if j - i <= 20:
        tmp = arr[i: j + 1]
        tmp.sort()
        arr[i: j + 1] = tmp
        return
    i_, j_ = i, j
    pivot = arr[i]
    while i != j:
        while i < j and arr[j] >= pivot:
            j -= 1
        if i < j:
            i += 1
        else:
            break
        while i < j and arr[i] < pivot:
            i += 1
        arr[i], arr[j] = arr[j], arr[i]
    arr[i_], arr[i] = arr[i], arr[i_]
    sort(i_, i - 1, arr)
    sort(i + 1, j_, arr)


# random.seed(123)
test_arr = random_list(10000, 0, 100000)
# test_arr = [2, 1, 0]
# test_arr = [i for i in range(10000)]
print(test_arr)
start = time.time()
sort(0, len(test_arr) - 1, test_arr)
end = time.time()
print(test_arr)
print('%.2fs' % (end - start))
print(is_sort(test_arr))
