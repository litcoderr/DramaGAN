
import json


def parse_json(file_path):
    with open(file_path, 'r') as file:
        parsed = json.load(file)
    return parsed
