from typing import Dict, Optional, Tuple, Union

from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import Regularizer

IDVocab = Dict[str, int]
RatingTriplet = Tuple[int, int, int]

ActivationType = Optional[Union[Activation, str]]
InitializerType = Optional[Union[Initializer, str]]
RegularizerType = Optional[Union[Regularizer, str]]
