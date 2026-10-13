"""Opt-in CPU staging for two logical ranks sharing one CUDA device."""


class StagedWork:
    def __init__(self, work, staging, destination=None):
        self._work = work
        self._staging = staging
        self._destination = destination
        self._copied = False





def install_cpu_staging(dist):
    """Make Gloo point-to-point calls accept existing CUDA call sites."""
    if getattr(dist, "_ssd_cpu_staging_installed", False):
        return
    original_send = dist.send
    original_recv = dist.recv
    original_isend = dist.isend
    original_irecv = dist.irecv



    def isend(tensor, *args, **kwargs):
        staged = tensor.detach().to(device="cpu") if tensor.is_cuda else tensor
        return StagedWork(original_isend(staged, *args, **kwargs), staged)


    dist.send = send
    dist.recv = recv
    dist.isend = isend
    dist.irecv = irecv
    dist._ssd_cpu_staging_installed = True
