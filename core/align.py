import os
import tensorflow as tf

class Align(object):
    def __init__(self, align_len=40, drop_first_token=False):
        self.align_len = align_len
        self.drop_first_token = drop_first_token
        # Define vocabulary.
        self.vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
    
        self.char_to_num = tf.keras.layers.StringLookup(
            vocabulary=self.vocab, oov_token=""
        )
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )
    
    def get_vocabulary(self):
        return self.vocab
    
    # Function to load the alignments of index correspondence
    def load_alignments(self, path:str) -> tf.Tensor: 
        if not os.path.exists(path):
            return tf.zeros([self.align_len], dtype=tf.int64)
        
        with open(path, 'r') as f: 
            lines = f.readlines()

        tokens = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and parts[2] != 'sil': 
                tokens.append(parts[2])
        
        if not tokens:
            print(f"Warning: No valid tokens found in {path}.")
            return tf.zeros([self.align_len], dtype=tf.int64)
        
        joined = "".join(tokens)
        # Convert the string into a list of characters
        tokens_list = list(joined)

        # Optionally drop the first token
        if self.drop_first_token and len(tokens_list) > 0:
            tokens_list = tokens_list[1:]

        # Ensure the list has exactly self.align_len elements:
        if len(tokens_list) > self.align_len:
            tokens_list = tokens_list[:self.align_len]
        else:
            # Pad with a space if too short.
            tokens_list += [" "] * (self.align_len - len(tokens_list))

        # Convert list to a tf.Tensor
        tokens_tensor = tf.constant(tokens_list)
        # Conver characters to numeric indices
        numeric = self.char_to_num(tokens_tensor) - 1

        numeric.set_shape([self.align_len]) # Explicitly set the static shape.
        return numeric