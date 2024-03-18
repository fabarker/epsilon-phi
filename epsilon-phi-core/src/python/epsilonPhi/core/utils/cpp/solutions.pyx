from typing import List

def count_increases(depths: List[int]) -> int:
    current_depth = depths[0]
    increase_counter: int = 0
    for depth in depths[1:]:
        if depth > current_depth:
            increase_counter += 1
        current_depth = depth
    return increase_counter