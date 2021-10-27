# 按low进行划分
def partition(arr, low, high):
    # 划分
    pivot = arr[low]
    while (low < high):
        while low < high and pivot <= arr[high]:
            high -= 1
        arr[low] = arr[high]
        while low < high and arr[low] <= pivot:
            low += 1
        arr[high] = arr[low]

    arr[low] = pivot
    return low


# 快排[low, high] 闭区间
def quick_sort(arr, low, high):
    if low < high:
        # 用mid进行划分(避免顺序或逆序时最坏复杂度)
        mid = (low + high) // 2
        arr[low], arr[mid] = arr[mid], arr[low]
        pivot_pos = partition(arr, low, high)
        quick_sort(arr, low, pivot_pos - 1)
        quick_sort(arr, pivot_pos + 1, high)


from random import randint
from time import clock

l = []
for i in range(100000):
    l.append(randint(0, 100000))
t = clock()
quick_sort(l, 0, 100000 - 1)  # python
print(clock() - t)
print(l[:100])

l = []
for i in range(100000):
    l.append(randint(0, 100000))
t = clock()
l.sort()  # python-stable
print(clock() - t)
print(l[:100])

from sort import quick_sort

l = []
for i in range(100000):
    l.append(randint(0, 100000))
t = clock()
quick_sort(l)  # cython
print(clock() - t)
print(l[:100])
