class Heap:
    def __init__(self, items: list=None):
        self.heap_list = [-999999]
        self.heap_size = 0
        if items:
            self.heap_list.extend(items)
            self.heap_size = len(self.heap_list) - 1
            self.build_heap()

    def insert(self, x):
        """
        percolate up
        :return:
        """
        if self.heap_size < len(self.heap_list) - 1:
            self.heap_list[self.heap_size + 1] = x
        else:
            self.heap_list.append(x)
        self.heap_size += 1
        hole = self.heap_size
        while self.heap_list[hole] < self.heap_list[hole // 2]:
            self.heap_list[hole // 2], self.heap_list[hole] = self.heap_list[hole], self.heap_list[hole // 2]
            hole //= 2

    def delete(self):
        deleted_item = self.heap_list[self.heap_size]
        self.heap_list[1], self.heap_list[self.heap_size] = self.heap_list[self.heap_size], self.heap_list[1]
        self.heap_size -= 1
        self.percolate_down(1)
        return deleted_item

    def percolate_down(self, hole):
        tmp = self.heap_list[hole]
        while hole * 2 <= self.heap_size:
            child = hole * 2
            if child + 1 <= self.heap_size and self.heap_list[child + 1] < self.heap_list[child]:
                child += 1
            # 堆的最后一个值是有可能比 child 小的，小就没必要交换了
            if tmp > self.heap_list[child]:
                self.heap_list[hole] = self.heap_list[child]
            else:
                break
            hole = child
        self.heap_list[hole] = tmp

    def build_heap(self):
        for i in range(self.heap_size // 2, 0, -1):
            self.percolate_down(i)


if __name__ == '__main__':
    heap = Heap([9, 14, 8, 46, 98, 23, 77, 34, 50, 12, 29, 44])
    print(heap.heap_list)
    heap.delete()
    print(heap.heap_list)
    heap.insert(33)
    print(heap.heap_list)
    heap.delete()
    print(heap.heap_list)




