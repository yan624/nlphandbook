import random


def random_list(n, min_, max_):
    return [random.randint(min_, max_) for _ in range(n)]


def is_sort(arr, ascend=True):
    for i in range(1, len(arr)):
        if ascend:
            if arr[i - 1] > arr[i]:
                return False
        else:
            if arr[i - 1] < arr[i]:
                return False
    return True



