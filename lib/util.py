
import json


class Config:
    def __init__(self, path: str):
        self.__dict__.update(load_json(path))

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

    def update(self, config: dict):
        self.__dict__.update(config)

    def __setitem__(self, key, value):
        self.__dict__.__setitem__(key, value)

    def __repr__(self):
        return str(self.__dict__)


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_data(path: str):
    with open(path, 'r') as f:
        result = f.readlines()
    return result