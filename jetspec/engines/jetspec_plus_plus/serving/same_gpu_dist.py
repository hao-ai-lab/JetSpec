"""Opt-in CPU staging for two logical ranks sharing one CUDA device."""


class StagedWork:
    def __init__(self, work, staging, destination=None):
        self._work = work
        self._staging = staging
        self._destination = destination
        self._copied = False

    def _finish_receive(self):
        if self._destination is not None and not self._copied:
            self._destination.copy_(self._staging)
            self._copied = True

    def wait(self, *args, **kwargs):
        result = self._work.wait(*args, **kwargs)
        self._finish_receive()
        return result

    def is_completed(self):
        completed = self._work.is_completed()
        if completed:
            self._finish_receive()
        return completed


def install_cpu_staging(dist):
    """Make Gloo point-to-point calls accept existing CUDA call sites."""
    if getattr(dist, "_ssd_cpu_staging_installed", False):
        return
    original_send = dist.send
    original_recv = dist.recv
    original_isend = dist.isend
    original_irecv = dist.irecv

    def send(tensor, *args, **kwargs):
        staged = tensor.detach().to(device="cpu") if tensor.is_cuda else tensor
        return original_send(staged, *args, **kwargs)

    def recv(tensor, *args, **kwargs):
        if not tensor.is_cuda:
            return original_recv(tensor, *args, **kwargs)
        staged = tensor.new_empty(tensor.shape, device="cpu")
        result = original_recv(staged, *args, **kwargs)
        tensor.copy_(staged)
        return result

    def isend(tensor, *args, **kwargs):
        staged = tensor.detach().to(device="cpu") if tensor.is_cuda else tensor
        return StagedWork(original_isend(staged, *args, **kwargs), staged)

    def irecv(tensor, *args, **kwargs):
        if not tensor.is_cuda:
            return original_irecv(tensor, *args, **kwargs)
        staged = tensor.new_empty(tensor.shape, device="cpu")
        return StagedWork(original_irecv(staged, *args, **kwargs), staged, tensor)

    dist.send = send
    dist.recv = recv
    dist.isend = isend
    dist.irecv = irecv
    dist._ssd_cpu_staging_installed = True
