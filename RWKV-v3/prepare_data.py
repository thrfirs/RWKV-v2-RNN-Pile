########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Process, Queue, Manager

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

input_file = "/root/ao3_crawl/concat.txt"
output_file = "/root/ao3_crawl/concat.npy"

TASK = 'tokenize' # tokenize verify
# TASK = "test"

NUM_WORKERS = 8


class TokenizerProcess(Process):
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
        self._input_queue = input_queue
        self._output_queue = output_queue
    
    def run(self):
        while True:
            item = self._input_queue.get()
            if item is None:
                break
            index, chunk = item
            data_code = self._tokenizer.encode(chunk)
            self._output_queue.put((index, np.array(data_code, dtype='uint16')))
            del chunk, data_code, item


if TASK == 'tokenize':
    print(f'Tokenizing {input_file} (VERY slow. please wait)')

    data_raw = open(input_file, encoding="utf-8").read()
    data_raw_length = len(data_raw)
    print(f'Raw length = {data_raw_length}')

    # tokenize in chunks and in parallel
    chunk_size = 1000000

    manager = Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()
    tokenizer_processes = [TokenizerProcess(input_queue, output_queue) for _ in range(NUM_WORKERS)]
    for process in tokenizer_processes:
        process.start()
    
    for i in trange(0, data_raw_length, chunk_size, desc="Dispatch", dynamic_ncols=True):
        input_queue.put((i, data_raw[i:i+chunk_size]))
    del data_raw
    for process in tokenizer_processes:
        input_queue.put(None)
    
    token_chunks = []
    for i in trange(0, data_raw_length, chunk_size, desc="Gather", dynamic_ncols=True):
        token_chunks.append(output_queue.get())
    token_chunks.sort(key=lambda x: x[0])
    all_tokens = np.concatenate([x[1] for x in token_chunks])
    tqdm.write(f'Total encoded length = {len(all_tokens)}')
    np.save(output_file, all_tokens, allow_pickle=False)
    
    for process in tokenizer_processes:
        process.join()

elif TASK == 'verify':
    test = np.load(output_file)
    print(test)
    print('\n\n')
    print(tokenizer.decode(test[:100]))
    print('\n\n')
    print(tokenizer.decode(test[-100:]))

elif TASK == "test":
    test_str = "测试1<|endoftext|>测试2<|endoftext|>测试3"
    test_result = tokenizer.encode(test_str)
    print(test_result)

