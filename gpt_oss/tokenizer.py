import tiktoken

def get_tokenizer():
    # Try o200k_base (newer tiktoken versions), fallback to cl100k_base
    try:
        base_encoding = tiktoken.get_encoding("o200k_base")
        base_vocab_size = 200000
    except ValueError:
        # Fallback for older tiktoken versions
        base_encoding = tiktoken.get_encoding("cl100k_base")
        base_vocab_size = 100000
        print("Warning: o200k_base not found, using cl100k_base fallback. Consider upgrading tiktoken.")
    
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=base_encoding._pat_str,
        mergeable_ranks=base_encoding._mergeable_ranks,
        special_tokens={
            **base_encoding._special_tokens,
            "<|startoftext|>": base_vocab_size - 2,
            "<|endoftext|>": base_vocab_size - 1,
            "<|reserved_200000|>": base_vocab_size,
            "<|reserved_200001|>": base_vocab_size + 1,
            "<|return|>": base_vocab_size + 2,
            "<|constrain|>": base_vocab_size + 3,
            "<|reserved_200004|>": base_vocab_size + 4,
            "<|channel|>": base_vocab_size + 5,
            "<|start|>": base_vocab_size + 6,
            "<|end|>": base_vocab_size + 7,
            "<|message|>": base_vocab_size + 8,
            "<|reserved_200009|>": base_vocab_size + 9,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {
            f"<|reserved_{i}|>": i for i in range(200013, 201088)
        },
    )
    return tokenizer
