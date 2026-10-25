"""Rank-local CPU placement for serving. Auto skips each NUMA node's first 12 CPUs."""
import os
from pathlib import Path
import re
import subprocess


def cpu_list(text):
    cpus = []
    for item in text.strip().split(","):
        ends = item.split("-")
        if not item or len(ends) > 2 or not all(end.isdecimal() for end in ends):
            raise ValueError(f"invalid CPU list: {text!r}")
        first, last = int(ends[0]), int(ends[-1])
        if first > last:
            raise ValueError(f"descending CPU list: {text!r}")
        cpus.extend(range(first, last + 1))
    if len(cpus) != len(set(cpus)):
        raise ValueError(f"overlapping CPU list: {text!r}")
    return sorted(cpus)


def parse_cpu_ranges(spec, ranks):
    groups = spec.split(";")
    if len(groups) != ranks:
        raise ValueError(f"SSD_RANK_CPU_AFFINITY needs {ranks} CPU range(s)")
    parsed = []
    for group in groups:
        if not re.fullmatch(r"[0-9]+-[0-9]+", group):
            raise ValueError("SSD_RANK_CPU_AFFINITY needs inclusive A-B ranges")
        try:
            cpus = cpu_list(group)
        except ValueError as error:
            raise ValueError(f"SSD_RANK_CPU_AFFINITY {error}") from error
        if any(set(cpus).intersection(previous) for previous in parsed):
            raise ValueError("SSD_RANK_CPU_AFFINITY ranges must be disjoint")
        parsed.append(cpus)
    return parsed


def gpu_pci_bus_id(uuid):
    bus = subprocess.check_output([
        "nvidia-smi", "-i", uuid, "--query-gpu=pci.bus_id", "--format=csv,noheader"],
        text=True, timeout=30).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bus):
        raise ValueError(f"invalid GPU PCI bus ID: {bus!r}")
    return bus[-12:]


def plan_cpu_pin(mode, gpu_uuids, placement, ranks, *, sysfs=Path("/sys"), pci_buses=None):
    """Choose equal contiguous blocks, in CPU-list order, after reserving 12 per node."""
    if mode not in ("auto", "off"):
        groups = parse_cpu_ranges(mode, ranks)
        nodes = {}
        for node_dir in (sysfs / "devices/system/node").glob("node[0-9]*"):
            nodes[int(node_dir.name[4:])] = set(cpu_list((node_dir / "cpulist").read_text()))
        entries = []
        for rank, group in enumerate(groups):
            local = [node for node, cpus in nodes.items() if set(group) <= cpus]
            entries.append(dict(rank=rank, gpu=gpu_uuids[min(rank, 1) if placement == "pair" else 0],
                                numa=local[0] if len(local) == 1 else None,
                                cpus=group, block_size=len(group)))
        return dict(mode="explicit", ranks=entries, spec=mode)
    if mode == "off":
        return dict(mode="off", ranks=[
            dict(rank=rank, gpu=gpu_uuids[min(rank, 1) if placement == "pair" else 0],
                 numa=None, cpus=[], block_size=0) for rank in range(ranks)], spec=None)
    if not gpu_uuids or any(not uuid for uuid in gpu_uuids):
        raise ValueError("CPU_PIN auto requires resolved GPU UUIDs")
    pci_buses = pci_buses or [gpu_pci_bus_id(uuid) for uuid in gpu_uuids]
    gpu_nodes = []
    for bus in pci_buses:
        path = sysfs / "bus/pci/devices" / bus / "numa_node"
        node = int(path.read_text().strip())
        if node < 0:
            raise ValueError(f"CPU_PIN GPU {bus} has no NUMA node")
        gpu_nodes.append(node)
    rank_nodes = [gpu_nodes[min(rank, 1) if placement == "pair" else 0] for rank in range(ranks)]
    entries = [None] * ranks
    for node in sorted(set(rank_nodes)):
        node_ranks = [rank for rank, local in enumerate(rank_nodes) if local == node]
        cpus = cpu_list((sysfs / f"devices/system/node/node{node}/cpulist").read_text())[12:]
        size = min(12, len(cpus) // len(node_ranks))
        if size < 1:
            raise ValueError(f"CPU_PIN NUMA node {node} too small after reserving 12 CPUs")
        blocks = []
        for start in range(len(cpus) - size + 1):
            block = cpus[start:start + size]
            if block[-1] - block[0] == size - 1 and not set(block).intersection(
                    cpu for previous in blocks for cpu in previous):
                blocks.append(block)
                if len(blocks) == len(node_ranks):
                    break
        if len(blocks) != len(node_ranks):
            raise ValueError(f"CPU_PIN NUMA node {node} lacks contiguous CPU blocks")
        for rank, block in zip(node_ranks, blocks):
            entries[rank] = dict(rank=rank, gpu=gpu_uuids[min(rank, 1) if placement == "pair" else 0],
                                 numa=node, cpus=block, block_size=size)
    return dict(mode="auto", ranks=entries,
                spec=";".join(f"{row['cpus'][0]}-{row['cpus'][-1]}" for row in entries))


def pin_rank_cpu_affinity(spec, rank, *, setter=None, plan=None):
    if plan is None and not spec:
        return
    groups = parse_cpu_ranges(spec, len(plan["ranks"]) if plan else (3 if spec.count(";") == 2 else 2))
    if rank not in range(len(groups)):
        raise ValueError("SSD_RANK_CPU_AFFINITY invalid rank")
    cpus = groups[rank]
    if plan is not None and cpus != plan["ranks"][rank]["cpus"]:
        raise ValueError("CPU_PIN child affinity differs from requested plan")
    (setter or os.sched_setaffinity)(0, set(cpus))
    if plan is None:
        print(f"SSD_RANK_CPU_AFFINITY rank={rank} cpus={spec.split(';')[rank]}", flush=True)
    else:
        row = plan["ranks"][rank]
        print(f"CPU_PIN rank={rank} gpu={row['gpu']} numa={row['numa']} "
              f"cpus={','.join(map(str, cpus))}", flush=True)
