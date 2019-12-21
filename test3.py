from deepgauge_cov import get_boundary

tr_n_v = [[1, 5, 3], [7, 2, 6], [4, 8, 9]]
ts_n_v = [[0, 10, 3], [5, 1, 5], [10, 0, 8], [12, 5, 0]]
bound = get_boundary(tr_n_v, 3)
print(bound)
