from abc import ABC, abstractmethod


class AbstractAttention(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

class AbstractTransformer(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


    