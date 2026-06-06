import os
import numpy as np

class BinaryWriter:
    def __init__(self, root_path):
        self.root_path = root_path
    
        os.makedirs(self.root_path, exist_ok=True)
    
        self.data_filepath = os.path.join(self.root_path, 'data.bin')
        self.offsets_path = os.path.join(self.root_path, 'offsets.npy')
        
        # Open the data file as a binary file write-only
        self.data_file = open(self.data_filepath, 'wb')
        self.offsets = []
        self.current_offset = 0
    
    def write(self, item_bytes: bytes) -> int:
        if not isinstance(item_bytes, bytes):
            raise TypeError("item_bytes must be of type bytes")
        
        item_start = self.current_offset
        self.data_file.write(item_bytes)
        item_end = self.data_file.tell()
        self.offsets.append((item_start, item_end))
        self.current_offset = item_end

    def save_offsets(self):
        np.save(self.offsets_path, np.array(self.offsets, dtype=np.int64))

    def close(self):
        if self.data_file:
            self.data_file.close()
            self.data_file = None
        self.save_offsets()

    def __del__(self):
        self.close()

class BinaryReader:
    def __init__(self, root_path, close_file_after_reading=False):
        
        self.root_path = root_path
        self.data_filepath = os.path.join(self.root_path, 'data.bin')
        self.offsets_path = os.path.join(self.root_path, 'offsets.npy')

        self.offsets = np.load(self.offsets_path)
        self.fp = None
        self.close_file_after_reading = close_file_after_reading

    def __getitem__(self, index):
        if index < 0 or index >= self.offsets.shape[0]:
            raise IndexError("Index out of bounds")
        
        start, end = self.offsets[index]

        if self.fp is None:
            self.fp = open(self.data_filepath, 'rb')
        
        self.fp.seek(start)
        item_bytes = self.fp.read(end - start)

        if self.close_file_after_reading:
            self.fp.close()
            self.fp = None
        
        return item_bytes

    def __len__(self):
        return self.offsets.shape[0]
    
    def __del__(self):
        if self.fp:
            self.fp.close()
            self.fp = None
