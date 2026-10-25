"""A captured fork/join for context projection independent of row selection."""
import torch


class DraftStreams:
    def __init__(self, device):
        self.stream = torch.cuda.Stream(device=device)
        self.fork = torch.cuda.Event()
        self.join = torch.cuda.Event()

    def start(self, body):
        """Read the caller's completed C1 tail on the context stream."""
        self.fork.record(torch.cuda.current_stream(self.stream.device))
        self.stream.wait_event(self.fork)
        with torch.cuda.stream(self.stream):
            body()

    def finish(self):
        """J0 and final row geometry must see the context projection writes."""
        self.join.record(self.stream)
        torch.cuda.current_stream(self.stream.device).wait_event(self.join)
