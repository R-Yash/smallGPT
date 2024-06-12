class BPETokenizer:
    VOCAB_SIZE = 276
    NUM_MERGES = VOCAB_SIZE - 256

    def __init__(self, file_path):
        """Initialize the BPETokenizer by reading the file and training the tokenizer."""
        self.text = self.read_file(file_path)
        self.toks = list(self.text.encode('utf-8'))
        self.merges = self.get_merges(self.toks)
        self.vocab_dict = self.initialize_vocab_dict(self.merges)

    def get_vocab_size(self):
        return self.VOCAB_SIZE
    
    def read_file(self, file_path):
        """Read the content of the file and return the text."""
        with open(file_path, 'r') as f:
            text = f.read()
        return text

    def get_stats(self, ids):
        """Calculate and return the frequency of adjacent pairs in ids."""
        pair_counts = {}
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def merge(self, ids, pair, idx):
        """Merge the most frequent pair in ids and return the new ids."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def get_merges(self, toks):
        """Perform the BPE merges on the tokens and return the merges dictionary."""
        merges = {}

        for i in range(self.NUM_MERGES):
            stats = self.get_stats(toks)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            toks = self.merge(toks, pair, idx)
            merges[pair] = idx
        return merges

    def initialize_vocab_dict(self, merges):
        """Initialize and return the vocabulary dictionary with the merges."""
        vocab_dict = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in merges.items():
            vocab_dict[idx] = vocab_dict[p0] + vocab_dict[p1]
        
        return vocab_dict

    def decode(self, ids):
        """Decode the ids using the vocabulary dictionary."""
        tokens = b"".join(self.vocab_dict[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text

    def encode(self, text):
        """Encode the text using the merges dictionary."""
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

# Only run the main function when this script is executed directly
if __name__ == "__main__":
    tokenizer = BPETokenizer('data.txt')
    encoded_text = tokenizer.encode("example text")
    decoded_text = tokenizer.decode(encoded_text)

    print("Encoded Text:", encoded_text)
    print("Decoded Text:", decoded_text)
