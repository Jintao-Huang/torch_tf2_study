template<typename T>
inline void _swap(T &a, T &b) {
    T c = a;
    a = b;
    b = c;
}

// 按low进行划分
template<typename T>
int partition(T arr[], int low, int high) {
    // 划分
    T pivot = arr[low];
    while (low < high) {
        while (low < high && pivot <= arr[high]) --high;
        arr[low] = arr[high];
        while (low < high && arr[low] <= pivot) ++low;
        arr[high] = arr[low];
    }
    arr[low] = pivot;
    return low;
}
// 快排[low, high] 闭区间
template<typename T>
void quick_sort(T arr[], int low, int high) {
    if (low < high) {
        // 用mid进行划分(避免顺序或逆序时最坏复杂度)
        int mid = (low + high) / 2;
        _swap(arr[low], arr[mid]);
        int pivot_pos = partition(arr, low, high);
        quick_sort(arr, low, pivot_pos - 1);
        quick_sort(arr, pivot_pos + 1, high);
    }
}