# Taken from https://github.com/noamraph/tqdm/blob/master/tqdm.py
class StatusPrinter(object):
    def __init__(self, file):
        self.file = file
        self.last_printed_len = 0
    
    def print_status(self, s):
        self.file.write('\r'+s+' '*max(self.last_printed_len-len(s), 0))
        self.file.flush()
        self.last_printed_len = len(s)