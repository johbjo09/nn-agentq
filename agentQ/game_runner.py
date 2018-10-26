import sys
import os
import subprocess
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

CMD = "./runner"

def run_game(args):
    uid, parameters = args

    str_parameters = str(len(parameters)) + " " + " ".join(map(str, parameters))

    attempts = 0

    while attempts < 5:
        attempts += 1
        try:
            proc = subprocess.run(CMD,
                                  input = str_parameters.encode('ascii'),
                                  stdout = subprocess.PIPE,
                                  stderr = subprocess.PIPE)
            output = proc.stdout.decode('ascii').rstrip()
            result = output.split(" ")
            if len(result) == 4:
                return uid, int(result[0]), int(result[1]), int(result[2]), result[3]
        except subprocess.TimeoutExpired:
            proc.kill()
            time.sleep(1)

class GameRunner():
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=(os.cpu_count()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        return False


    def run_batch(self, batch):
        params = [(snake.uid, snake.get_parameters()) for snake in batch]
        return self.executor.map(run_game, params)
