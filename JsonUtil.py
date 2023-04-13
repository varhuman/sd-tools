import json
import os

class JsonUtil:

    def save(json_path):
        with open(json_path, 'w') as f:
            json.dump()

    def load(json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
