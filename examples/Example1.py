import numpy as np

arrbool = np.full(9, True, dtype=bool)

arr = np.arange(1,10)

arr2 = np.random.random(9)

#ustawienie precyzji wypisywania to 3 znaków po przecinku
np.set_printoptions(precision=3)
print(arr * arr2)

arr3 = np.resize(arr2,(3,3))

arr4 = np.array([1,2,3,2,3,4,3,4,5,6])
arr5 = np.array([7,2,10,2,7,4,9,4,9,8])

print(arr4+arr5)
print(np.setdiff1d(arr4,arr5))

#setdiff1d służy do usuwania powtórzeń
arr6 = np.setdiff1d(arr4,arr5) * np.random.randint(1,9, 9).sum()
