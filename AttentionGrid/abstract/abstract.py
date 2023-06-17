from abc import ABC, abstractmethod

class AbstractAttention(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        """ Perform the forward pass of the attention mechanism. """
        pass

class AbstractEncoder(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        """ Perform the forward pass of the encoder. """
        pass

class AbstractDecoder(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        """ Perform the forward pass of the decoder. """
        pass

class AbstractTransformer(AbstractEncoder, AbstractDecoder):
    """ 
    An abstract transformer class that inherits from the abstract encoder and decoder.
    This setup allows the transformer to use the forward method from both the encoder and decoder.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """ Perform the forward pass of the transformer. """
        pass
