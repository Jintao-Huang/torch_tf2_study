# distutils: language=c++
from libcpp.vector cimport vector
cdef extern from "quick_sort.cpp":
    void _quick_sort "quick_sort"(int arr[], int low, int high)

def quick_sort(arr, int low, int high):
    cdef vector[int] temp = arr  # list
    _quick_sort(temp.data(), low, high)
    arr[:] = temp