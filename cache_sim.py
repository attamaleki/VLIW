import argparse
import time
import numpy as np
from itertools import product
from tqdm import tqdm


class Simulator(object):

    def __init__(self,cache_size,block_size,associativity,trace_Array):
        self.cache_size = None
        self.block_size = None
        self.associativity = None
        self.n_blocks = None
        self.n_sets = None
        self.nb_address = 32
        self.nb_index = None
        self.nb_offset = None
        self.nb_tag = None

        self.trace = {
                'address': None,
                'tag':  None,
                'index': None,
                'offset': None,
                'hit': None
                }
        
        self.cache_size=cache_size
        self.block_size = block_size
        self.associativity = associativity
        # Calculate n blocks = cache size / block size
        self.n_blocks = self.cache_size // self.block_size

        self.associativity = int(self.associativity)
        self.n_sets = self.n_blocks // self.associativity

        # Calculate memory address bit configurations
        self.nb_offset = np.log2(self.block_size).astype(int)
        self.nb_index = np.log2(self.n_sets).astype(int)
        self.nb_tag = self.nb_address - (self.nb_index + self.nb_offset)

        # Default block data struct
        self.block = {
                'valid': 0,
                'tag': None,
                'last_used': 0
                }
        
        # k sets with n blocks
        self.sets = {k: {n: self.block.copy() for n in range(self.associativity)} for k in range(self.n_sets)}
        """ Parse memory addresses from trace file """

        # Read memory addresses and convert from str -> int
        trace = trace_Array
        self.trace['address'] = np.array([int(address, 16) for address in trace], dtype=np.uint32)
        self.trace['n_accesses'] = len(self.trace['address'])
        self.trace['hit'] = [False] * self.trace['n_accesses']

        # Calculate tag, index, and offset values
        # offset = address & 2**offset bits - 1
        self.trace['offset'] = np.bitwise_and(self.trace['address'], 2**self.nb_offset - 1)

        # tag = (address >> index+offset bits) & 2**tag bits - 1
        self.trace['tag'] = np.right_shift(self.trace['address'], self.nb_index + self.nb_offset)
        self.trace['tag'] = np.bitwise_and(self.trace['tag'], 2**self.nb_tag - 1)

        # If n_sets=1 then no bits are allocated for the index
        if self.n_sets == 1:
            self.trace['index'] = [0] * len(self.trace['address'])
        # Else index = (address >> offset bits) & 2**index bits - 1
        else:
            self.trace['index'] = np.right_shift(self.trace['address'], self.nb_offset)
            self.trace['index'] = np.bitwise_and(self.trace['index'], 2**self.nb_index - 1)


    def simulate(self):
        """ Run simulation """

        # Loop for all addresses in trace
        for i in tqdm(range(self.trace['n_accesses'])):

            tag, set = self.trace['tag'][i], self.trace['index'][i]

            # Check for tag match. If match then hit
            match = next((block for block in self.sets[set] \
                          if self.sets[set][block]['tag'] == tag), None)

            if match is not None:
                self.sets[set][match]['last_used'] = time.time()
                self.trace['hit'][i] = True
                continue

            # Check if any block in set has no data yet (valid=0)
            # MISS
            unset = next((block for block in self.sets[set] \
                          if self.sets[set][block]['valid'] == 0), None)

            if unset is not None:
                self.sets[set][unset]['valid'] = 1
                self.sets[set][unset]['tag'] = tag
                self.sets[set][unset]['last_used'] = time.time()
                continue

            # If all blocks contains data with different tags
            # Then replace via least recently used via the last_used param
            # MISS
            lru = np.argmin([self.sets[set][block]['last_used'] for block in self.sets[set]])

            self.sets[set][lru]['tag'] = tag
            self.sets[set][lru]['last_used'] = time.time()


        # Print results
        n_accesses = self.trace['n_accesses']
        hits = sum(self.trace['hit'])
        misses = n_accesses - hits
        hit_rate = (hits / n_accesses) * 100.0


        print('Cache Size {:d} - Block Size {:d} - Associativity {:d}'.format(self.cache_size, self.block_size, self.associativity))
        print('Num Blocks {:d} - Num Sets {:d}'.format(self.n_blocks, self.n_sets))
        print('Tag Length {:d} - Index Length {:d} - Offset Length {:d}'.format(self.nb_tag, self.nb_index, self.nb_offset))
        print('Num Accesses {:d} - Num Hits {:d} - Num Misses {:d}'.format(n_accesses, hits, misses))
        print('Hit Rate: {:.2f}'.format(hit_rate))
        return hit_rate

Trace=[
'0x018ADB20',
'0x018ADB28',
'0x01A5DB58',
'0x01A5DB50',
'0x01A5DB48',
'0x01A5DB40',
'0x01A5DB78',
'0x01A5DB70',
'0x01A5DB68',
'0x01A5DB60',
'0x0196CF98',
'0x0196CF90',
'0x0196CF88',
'0x0196CF80',
'0x014957D8',
'0x014957D0',
'0x014957C8',
'0x014957C0',
'0x014957F8',
'0x014957F0',
'0x014957E8',
'0x014957E0',
]
sim = Simulator(32768,64,4,Trace)
hit_rate = sim.simulate()



   
            
    
            
            


