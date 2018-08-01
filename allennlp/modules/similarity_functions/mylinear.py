import math

from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.common import Params
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.nn import Activation, util
import logging
logger = logging.getLogger('brc')


@SimilarityFunction.register("mylinear")
class MyLinearSimilarity(SimilarityFunction):
    """
    This similarity function performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.

    If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
    ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give ``x,y,x*y`` as the ``combination`` parameter to this class.  The computed similarity
    function would then be ``w^T [x; y; x*y] + b``, where ``w`` is a vector of weights, ``b`` is a
    bias parameter, and ``[;]`` is vector concatenation.

    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.

    Parameters
    ----------
    tensor_1_dim : ``int``
        The dimension of the first tensor, ``x``, described above.  This is ``x.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : ``int``
        The dimension of the second tensor, ``y``, described above.  This is ``y.size()[-1]`` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : ``str``, optional (default="x,y")
        Described above.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``w^T * [x;y] + b`` calculation.  Default is no
        activation.
    """
    def __init__(self,
                 tensor_1_dim: int,
                 tensor_2_dim: int,
                 combination: str = 'x,y',
                 activation: Activation = Activation.by_name('linear')()) -> None:
        super(MyLinearSimilarity, self).__init__()
        self._combination = combination
        self._part = len(self._combination.split(','))
        combined_dim = util.get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    @overrides
    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        '''
            tensor_1: [batch, len_1, d]
            tensor_2: [batch, len_2, d]
            self._weight_vector [3d]
            self._bias [3d]
        '''
        assert tensor_1.shape[-1] * self._part == self._weight_vector.shape[-1]
        d = self._weight_vector.shape[-1] // self._part
        assert self._weight_vector.shape[-1] % self._part == 0, '{} vs {}'.format(self._weight_vector.shape, self._part)
        sim_1 = torch.matmul(tensor_1, self._weight_vector[:d])
        sim_2 = torch.matmul(tensor_2, self._weight_vector[d:2*d])
        sim_1 = sim_1.unsqueeze(dim=2) # shape [batch, len_1, 1]
        sim_2 = sim_2.unsqueeze(dim=1) # shape [batch, 1, len_2]
        sim_1_temp = tensor_1 * self._weight_vector[2*d:3*d]
        sim_2_temp = tensor_2.permute(0,2,1).contiguous()
        sim_1_2 = torch.matmul(sim_1_temp, sim_2_temp)
        # shape [batch, len_1, len_2]
        similarity = sim_1 + sim_2 + sim_1_2
        return similarity

    @classmethod
    def from_params(cls, params: Params) -> 'MyLinearSimilarity':
        tensor_1_dim = params.pop_int("tensor_1_dim")
        tensor_2_dim = params.pop_int("tensor_2_dim")
        combination = params.pop("combination", "x,y")
        activation = Activation.by_name(params.pop("activation", "linear"))()
        params.assert_empty(cls.__name__)
        return cls(tensor_1_dim=tensor_1_dim,
                   tensor_2_dim=tensor_2_dim,
                   combination=combination,
                   activation=activation)
