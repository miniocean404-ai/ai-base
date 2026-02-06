import os


def create_dir(path: str):
    if not path.exists(path.dirname(path)):
        os.makedirs(path.dirname(path))
