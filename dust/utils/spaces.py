import torch


class Box:
    def __init__(self, dim, low=None, high=None, dtype=torch.float):
        """Constructor for Space.

        :param dim: Number of dimension in of the Box.
        :type dim: int
        :param low: Lower bounds of the Box space.
        :type low: float or list or torch.Tensor
        :param high: Higher bounds of the Box space.
        :type high: float or list or torch.Tensor
        :param dtype: Data type used in the Space, must be explicitly provided.
        :type dtype: torch.dtype
        """
        assert dtype is not None, "Data type must be explicitly provided."
        assert isinstance(
            dtype, torch.dtype
        ), "Data type must be of class `torch.dtype`."
        self.dtype = dtype

        assert dim > 0, "Dimension must be a strictly positive integer."
        self.__dim = dim
        self.__shape = torch.Size([self.dim])

        if low is None:
            self.low = torch.full(self.shape, -float("Inf"), dtype=torch.float)
        elif torch.as_tensor(low).ndim == 0:  # if scalar
            self.low = torch.full(self.shape, low, dtype=torch.float)
        else:
            assert (
                low.shape == self.shape
            ), "Lower boundary must have same dimensions as space Box."
            self.low = torch.as_tensor(low)
        if high is None:
            self.high = torch.full(self.shape, float("Inf"), dtype=torch.float)
        elif torch.as_tensor(high).ndim == 0:
            self.high = torch.full(self.shape, high, dtype=torch.float)
        else:
            assert (
                high.shape == self.shape
            ), "Higher boundary must have same dimensions as space Box."
            self.high = torch.as_tensor(high)

    @property
    def dim(self):
        """Returns the dimension of the Box.

        .. note::
            This is **not** equivalent to the tensor dimension.

        :return: The number of dimensions in the Box.
        :rtype: int
        """
        return self.__dim

    @property
    def shape(self):
        """Returns the shape of the Box.

        :return: The PyTorch dimensions of the Box.
        :rtype: torch.Size
        """
        return self.__shape
