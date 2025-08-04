import mmh3
import itertools
from bitarray import bitarray

def exact_deduplication():
    # Original items
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items
    # Compute hash -> list of items with that hash
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)
    # Keep one item from each group
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items
    print(deduped_items)
    
    
def build_table(items: list[str], num_bins: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        h = mmh3.hash(item) % num_bins  # @inspect item, @inspect h
        table[h] = 1  # @inspect table
    return table

def build_table_k(items: list[str], num_bins: int, k: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it.
    Use `k` hash functions."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        # For each of the k functions
        for seed in range(k):
            h = mmh3.hash(item, seed) % num_bins  # @inspect item, @inspect h, @inspect seed
            table[h] = 1  # @inspect table
    return table

def query_table(table: bitarray, item: str, num_bins: int, seed: int = 0):
    """Return whether `item` is in the `table`."""
    h = mmh3.hash(item, seed) % num_bins
    return table[h]

    
def query_table_k(table: bitarray, item: str, num_bins: int, k: int):
    """Return 1 if table set to 1 for all `k` hash functions."""
    return int(all(
        query_table(table, item, num_bins, seed)
        for seed in range(k)
    ))


def bloom_filter():
    items = ["the", "cat", "in", "the", "hat"]
    non_items = ["what", "who", "why", "when", "where", "which", "how"]
    
    # First, make the range of hash function small (small number of bins).
    m = 8  # Number of bins
    table = build_table(items, m)
    for item in items:
        assert query_table(table, item, m) == 1
    result = {item: query_table(table, item, m) for item in non_items}  # @inspect result
    num_mistakes = sum(result.values())  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    print(f"False positive rate: {false_positive_rate:.3f}")


    # 改进一下，用k个hash函数
    k = 2  # Number of hash functions
    table = build_table_k(items, m, k)
    for item in items:
        assert query_table_k(table, item, m, k) == 1
    result = {item: query_table_k(table, item, m, k) for item in non_items}  # @inspect result
    num_mistakes = sum(result.values())  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    print(f"False positive rate of k={k}: {false_positive_rate:.3f}")


def minhash(S: set[str], k: int):
    """Return a list of `k` hash values for the set `S`."""
    return  min(mmh3.hash(x, k) for x in S)


def compute_jaccard(A, B):
    intersection = len(A & B)  # @inspect intersection
    union = len(A | B)  # @inspect union
    return intersection / union


def test_jaccard_minhash():
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}
    jaccard = compute_jaccard(A, B)
    
    n = 100
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
    estimated_jaccard = sum(matches) / len(matches)

    assert abs(estimated_jaccard - jaccard) < 0.01


if __name__ == "__main__":
    # test_jaccard_minhash()
    bloom_filter()
    


