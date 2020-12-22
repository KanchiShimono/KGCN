import pickle
from typing import Dict

from pythonjsonlogger.jsonlogger import JsonFormatter


def get_json_formatter() -> JsonFormatter:
    return JsonFormatter(
        '%(asctime)s %(filename)s %(lineno)d %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')


def deserialize_pickle(path: str) -> Dict[str, int]:
    with open(path, 'rb') as f:
        return pickle.load(f)
