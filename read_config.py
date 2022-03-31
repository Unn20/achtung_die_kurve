import json

def read_config(path="config.json"):
    config = {}
    with open(path) as f:
        config = {**config, **json.load(f)}
    return config
