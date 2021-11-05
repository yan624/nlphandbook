import random
import time

from utils import random_list, is_sort
from tree.heap import Heap


# random.seed(123)
test_arr = random_list(10000, 0, 100000)
# test_arr = [i for i in range(10000)]
print(test_arr)
start = time.time()
heap = Heap(test_arr)
for _ in range(heap.heap_size):
    heap.delete()
end = time.time()
print(heap.heap_list)
print(heap.heap_size)
print(list(reversed(heap.heap_list))[:-1])
print('%.2fs' % (end - start))
print(is_sort(list(reversed(heap.heap_list))[:-1]))
