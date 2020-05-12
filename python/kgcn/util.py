from pythonjsonlogger.jsonlogger import JsonFormatter


def get_json_formatter() -> JsonFormatter:
    return JsonFormatter(
        '%(asctime)s %(filename)s %(lineno)d %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
