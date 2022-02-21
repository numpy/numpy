from numpy.core import inf, int32, zeros


class MatmulChain:
    """
    This class provides a optimization of a chain of matrix multiplication
    using dynamic programming.
    Using `MatmulChain([A, B, C, D]).get()` instead of `A @ B @ C @ D` 
    will probably improve performance.
    """

    def __init__(self, *matrices):
        """
        Initialize the chain with matrices to be multiplied.

        Parameters
        ----------
        matrices : tuple[matrix]
            The matrices to operate the multiplication.

        Raises
        ------
        ValueError
            It's raised if there is a pair of adjacent matrices that can't be
            multiplied.
        """
        self._matrices = list(matrices)
        self._shape = (
            None
            if len(matrices) == 0
            else (matrices[0].shape[: -2] +
                  [matrices[0].shape[0], matrices[-1].shape[1]])
        )

        for i in range(1, len(matrices)):
            if (matrices[i - 1].shape[1] != matrices[i].shape[0] or
                    matrices[i - 1].shape[: -2] != matrices[i].shape[: -2]):
                raise ValueError(
                    f"The matrix of index {i-1} is not left-multipliable "
                    "to that of index {i}"
                )

    @property
    def shape(self):
        """
        The shape of result. This operation doesn't execute the 
        multiplication, but it simply gives the result according 
        to the multipliers' shape. If no matrices were added to the
        chain, it'll provide `None`.
        """
        return tuple(self._shape)

    def __imatmul__(self, next):
        """
        Add a matrix to the chain.

        Parameters
        ----------
        next : array_like
            The matrix to be added.

        Raises
        ------
        ValueError
            It's raised if `next` isn't right-multipliable to the chain
        """
        if self._shape is not None and self._shape[1] != next.shape[0]:
            raise ValueError(
                f"The matrix is not right-multipliable to the chain")

        if isinstance(next, MatmulChain):
            self._matrices += next._matrices
        else:
            self._matrices.append(next)
        self._shape[-1] = next.shape[-1]
        return self

    def get(self):
        """Returns the result of multiplication in an optimized way"""
        splitPoint = self._optimize()
        return self._get(self._matrices,
                         0, len(self._matrices) - 1,
                         splitPoint)

    def _optimize(self):
        matrices = self._matrices
        cost = zeros([len(matrices), len(matrices)])
        splitPoint = zeros([len(matrices), len(matrices)], dtype=int32)
        for i in range(len(matrices) - 1, -1, -1):
            cost[i, i] = 0
            splitPoint[i, i] = i
            iSize = matrices[i].shape[-2]
            for j in range(i + 1, len(matrices)):
                jSize = matrices[j].shape[1]
                minK = None
                currentMinCost = inf
                for k in range(i, j):
                    kSize = matrices[k].shape[-1]
                    currentCost = (cost[i, k] + cost[k + 1, j] +
                                   iSize * kSize * jSize)
                    if currentCost < currentMinCost:
                        currentMinCost = currentCost
                        minK = k
                cost[i, j] = currentMinCost
                splitPoint[i, j] = minK
        return splitPoint

    def _get(self, matrices, i, j, splitPoint):
        if i == j:
            return matrices[i]
        elif i + 1 == j:
            return matrices[i] @ matrices[j]
        k = splitPoint[i, j]
        left = self._get(matrices, i, k, splitPoint)
        right = self._get(matrices, k + 1, j, splitPoint)
        return left @ right
