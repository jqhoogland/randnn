import os, hashlib, logging
from typing import Optional

import numpy as np


def np_cache(dir_path: str = "./saves/", file_prefix: Optional[str] = None):
    """
    A wrapper to load a previous response to a function (or to run the function and save the result otherwise).
    Assuming the function returns a np.ndarray as its response
    """
    def inner(func):
        def wrapper(*args, **kwargs):
            params = (str(args) + str(kwargs)).encode('utf-8')
            file_name = file_prefix + hashlib.md5(params).hexdigest() + ".npy"
            file_path = os.path.join(dir_path, file_name)

            if not os.path.isdir(dir_path):
                logging.info("Creating directory %s", dir_path)
                os.mkdir(dir_path)

            if os.path.isfile(file_path):
                logging.info("Loading from save %s", file_path)

                return np.load(file_path)

            response = func(*args, **kwargs)

            logging.info("Saving to %s", file_path)
            np.save(file_path, response)
            return response

        return wrapper

    return inner
