from abc import ABC, abstractmethod


class Data(ABC):

    @abstractmethod
    def get_training(self):
        pass

    @abstractmethod
    def get_test(self):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_headers(self):
        pass
