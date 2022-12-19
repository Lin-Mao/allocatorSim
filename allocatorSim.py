#!/usr/bin/env python
import os
import re
import copy
from enum import Enum

max_block_size = 0


class Config(Enum):
    kRoundLarge = 2097152  # 2 MB
    kLargeBuffer = 20971520  # 20 MB
    kSmallSize = 1048576  # 1 MB


class AllocatorSim:
    def __init__(self):
        self.max_reserved_size = 0
        self.seg_id = 0
        self.segments = dict()  # {seg_id: [block, ], ...} block: [start, end, size, die] die=-1 means unused
        self.block_pool = list()  # [(seg_id, size), ...]

    def __del__(self):
        print("[Allocator report]")
        print(self.segments)
        print(self.block_pool)
        print("Number of large segments: {}".format(len(self.segments)))
        print("Max reserved size: {} B ({})".format(
            self.max_reserved_size, self.format_size(self.max_reserved_size)))

    def test_allocator(self):
        print("<<<<<<<<<<<<<<<<<<<<<<< Begin >>>>>>>>>>>>>>>>>>>>>>>")
        print("=================== block1 malloc ===================")
        block1 = self.block_malloc(Config.kLargeBuffer.value * 5, 10)
        print(self.segments)

        print("==================== block1 free ====================")
        self.block_free(block1[0], block1[1], block1[2])
        print(self.segments)

        print("=================== block2 malloc ===================")
        block2 = self.block_malloc(Config.kLargeBuffer.value * 1, 12)
        print(self.segments)

        print("=================== block3 malloc ===================")
        block3 = self.block_malloc(Config.kLargeBuffer.value * 2, 17)
        print(self.segments)

        print("=================== block4 malloc ===================")
        block4 = self.block_malloc(Config.kLargeBuffer.value * 1, 25)
        print(self.segments)

        print("==================== block4 free ====================")
        self.block_free(block4[0], block4[1], block4[2])
        print(self.segments)

        print("==================== block2 free ====================")
        self.block_free(block2[0], block2[1], block2[2])
        print(self.segments)

        print("==================== block3 free ====================")
        self.block_free(block3[0], block3[1], block3[2])
        print(self.segments)

        print("<<<<<<<<<<<<<<<<<<<<<<<< End >>>>>>>>>>>>>>>>>>>>>>>>")

    def dump_segments(self):
        for s in self.segments.keys():
            print(s, self.segments[s])

    def dump_block_pool(self):
        for p in self.block_pool:
            print(p)

    @staticmethod
    def format_size(size: int) -> str:
        if size <= 1024:
            return "{} B".format(size)
        elif size <= 1048576:
            return "{:.2f} KB".format(size / 1024)
        elif size <= 1073741824:
            return "{:.2f} MB".format(size / 1048576)
        else:
            return "{:.2f} GB".format(size / 1073741824)

    @staticmethod
    def default_get_allocation_size(size: int) -> int:
        kRoundLarge = Config.kRoundLarge.value
        return kRoundLarge * int((size + kRoundLarge - 1) / kRoundLarge)

    @staticmethod
    def customized_get_allocation_size1(size: int) -> int:
        count = 0
        b_size = size
        while b_size > 0:
            b_size = int(b_size / 2)
            count += 1
        return pow(2, count)

    @staticmethod
    def customized_get_allocation_size2(size: int) -> int:
        global max_block_size
        return max_block_size

    def get_allocation_size(self, size: int) -> int:
        return self.customized_get_allocation_size2(size)

    @staticmethod
    def lower_bound(nums: list, target: int) -> int:
        if not nums:
            return -1
        low, high = 0, len(nums) - 1
        pos = len(nums)
        while low < high:
            mid = int((low + high) / 2)
            if nums[mid][1] < target:
                low = mid + 1
            else:  # >=
                high = mid
                # pos = high
        if nums[low][1] >= target:
            pos = low
        return pos

    def update_block_pool(self):
        pass

    def update_segments(self):
        pass

    def get_free_block(self, size: int, die: int) -> (bool, int, int, int):
        self.block_pool = sorted(self.block_pool, key=lambda x: x[1], reverse=False)
        pos = self.lower_bound(self.block_pool, size)
        if pos == -1:
            return False, -1, -1, -1
        if pos == len(self.block_pool):
            return False, -1, -1, -1
        else:
            self.update_block_pool()
            seg_id = self.block_pool[pos][0]
            start = -1
            block_size = -1
            count = -1
            for s in self.segments[seg_id]:
                count += 1
                if s[2] == self.block_pool[pos][1] and s[3] == -1:
                    start = s[0]
                    block_size = s[2]
                    break

            # update pool and segments
            block = self.block_pool[pos]
            self.segments[seg_id][count][3] = die
            self.block_pool.remove(block)
            return True, seg_id, start, block_size

    def alloc_block(self, alloc_size: int, die: int) -> (bool, int, int, int):
        self.seg_id += 1
        self.max_reserved_size += alloc_size
        self.segments[self.seg_id] = [[0, alloc_size, alloc_size, die]]
        return True, self.seg_id, 0, alloc_size

    def release_available_cached_blocks(self):
        pass

    def release_cached_blocks(self):
        pass

    @staticmethod
    def should_split(size: int, block_size: int) -> bool:
        if block_size - size > Config.kLargeBuffer.value:
        # if block_size - size > Config.kSmallSize.value:
            return True
        else:
            return False

    def split_block(self, size: int, segment_id: int, start: int, block_size: int):
        count = 0
        die = -1
        end = -1
        for s in self.segments[segment_id]:
            if s[0] == start and s[2] == block_size:
                die = s[3]
                end = s[1]
                break
            count += 1

        # block: (start, end, size, die)
        self.segments[segment_id].remove(self.segments[segment_id][count])
        self.segments[segment_id].append([start, start + size, size, die])
        self.segments[segment_id].append([start + size, end, block_size - size, -1])

        self.block_pool.append((segment_id, block_size - size))

    def block_malloc(self, size: int, die: int) -> (int, int, int):
        alloc_size = self.get_allocation_size(size)

        block_found, segment_id, start, block_size = self.get_free_block(size, die)
        if block_found:
            pass
        else:
            # recycle segment
            block_found, segment_id, start, block_size = self.get_free_block(size, die)
            if block_found:
                pass
            else:
                block_found, segment_id, start, block_size = self.alloc_block(alloc_size, die)
                if block_found:
                    pass
                else:
                    self.release_available_cached_blocks()
                    self.release_cached_blocks()

        if not block_found:
            print("OOM!!")
            return -1, -1, -1

        if self.should_split(size, block_size):
            self.split_block(size, segment_id, start, block_size)
        else:
            size = block_size

        return segment_id, start, size

    def block_free(self, segment_id: int, start: int, size: int):
        # block: (start, end, size, die)
        count = 0
        end = start + size
        target = 0
        predecessor = -1
        successor = -1

        segment_list = copy.deepcopy(self.segments[segment_id])
        for s in segment_list:
            if s[0] == start and s[2] == size:
                target = count
            if s[1] == start and s[3] == -1:
                predecessor = count
            if s[0] == end and s[3] == -1:
                successor = count
            count += 1

        if predecessor != -1:
            start = segment_list[predecessor][0]
            size = size + segment_list[predecessor][2]
            self.block_pool.remove((segment_id, segment_list[predecessor][2]))
            self.segments[segment_id].remove(segment_list[predecessor])
        if successor != -1:
            end = segment_list[successor][1]
            size = size + segment_list[successor][2]
            self.block_pool.remove((segment_id, segment_list[successor][2]))
            self.segments[segment_id].remove(segment_list[successor])

        if predecessor != -1 or successor != -1:
            self.segments[segment_id].remove(segment_list[target])
            self.segments[segment_id].append([start, end, size, -1])
            self.block_pool.append((segment_id, size))
        else:
            self.segments[segment_id][target][3] = -1
            self.block_pool.append((segment_id, size))


def get_large_blocks(path: str) -> dict:
    large_blocks = dict()  # {op_id: size, ...}
    file = open(os.path.join(path, "submemory_size_list.txt"))
    line = file.readline()
    while line:
        nums = re.findall(r"\d+\.?\d*", line)
        if int(nums[1]) > 20971520:
            large_blocks[int(nums[0])] = int(nums[1])
        line = file.readline()
    file.close()
    # print(len(large_blocks))
    global max_block_size
    max_block_size = max([i for i in large_blocks.values()])
    return large_blocks


def parse_operation_map(path: str, large_blocks: dict) -> dict:
    operation_dict = dict()  # {op_id: [(op_id, op_type), ...]}
    file = open(os.path.join(path, "submemory_liveness.txt"))
    line = file.readline()
    while line:
        nums = re.findall(r"\d+\.?\d*", line)
        if not int(nums[0]) in large_blocks.keys():
            line = file.readline()
            continue
        temp = []
        for i in range(1, len(nums), 2):
            temp.append((int(nums[i]), int(nums[i + 1])))
        # print(temp)
        operation_dict[int(nums[0])] = temp
        line = file.readline()
    file.close()
    # print(len(operation_dict))
    return operation_dict


def parse_block_sequence(large_blocks: dict, operation_dict: dict) -> dict:
    block_sequence = dict()  # {op_id: (size, born, die), ...}
    keys = list(large_blocks.keys())
    keys.sort()
    for i in keys:
        begin = operation_dict[i][0][0]
        end = operation_dict[i][len(operation_dict[i]) - 1][0]
        block_sequence[i] = (large_blocks[i], begin, end)

    # print(len(block_sequence))
    return block_sequence


def allocator_sim(block_sequence: dict):
    offset = min([i[1] for i in block_sequence.values()])
    max_op = max([i[2] for i in block_sequence.values()]) + 1
    a_s = AllocatorSim()
    # a_s.test_allocator()

    dying_block = dict()
    for k in block_sequence.keys():
        dying_block[block_sequence[k][2] - offset] = k - offset

    block_map = dict()
    for i in range(offset, max_op):
        if i in block_sequence.keys():
            segment_id, start, size = a_s.block_malloc(block_sequence[i][0], block_sequence[i][2] - offset)
            block_map[i - offset] = (segment_id, start, size)
        if i - offset in dying_block.keys():
            db = dying_block[i - offset]
            a_s.block_free(block_map[db][0], block_map[db][1], block_map[db][2])


def main(path):
    large_blocks = get_large_blocks(path)
    operation_dict = parse_operation_map(path, large_blocks)
    blocks = parse_block_sequence(large_blocks, operation_dict)
    allocator_sim(blocks)


if __name__ == "__main__":
    input_path = "C:/Users/mao/Desktop/liveness/memory_liveness"
    main(input_path)
