"""J<m>-R<n> schedules, starting at J0 and alternating refresh and Jacobi passes."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Schedule:
    j: int
    r: int

    def __post_init__(self):
        if type(self.j) is not int or type(self.r) is not int or min(self.j, self.r) < 0:
            raise ValueError("J and R must be nonnegative integers")

    @classmethod
    def parse(cls, config):
        match = re.fullmatch(r'J([0-9]+)-R([0-9]+)', config or '')
        if match is None:
            raise ValueError(f'JETSPEC_CHAIN_FIXED_CONFIG={config!r}: expected J<m>-R<n>')
        return cls(*map(int, match.groups()))

    def __str__(self):
        return f"J{self.j}-R{self.r}"

    @property
    def operations(self):
        operations = ['J0']
        for index in range(1, max(self.j, self.r) + 1):
            if index <= self.r:
                operations.append(f'R{index}')
            if index <= self.j:
                operations.append(f'J{index}')
        return tuple(operations)

    @property
    def refresh_at(self):
        return tuple(range(1, self.r + 1))

    @property
    def stages(self):
        """A refresh and its following pass share a checkpoint."""
        stages = []
        for operation in self.operations:
            if operation.startswith('J') and stages and stages[-1][-1].startswith('R'):
                stages[-1] += (operation,)
            else:
                stages.append((operation,))
        return tuple(stages)

    @staticmethod
    def reachable(max_j, max_r):
        """Configurations in an inclusive bounded capture inventory."""
        Schedule(max_j, max_r)
        return tuple(str(Schedule(j, r)) for j in range(max_j + 1)
                     for r in range(max_r + 1))


parse_schedule = Schedule.parse


def schedule_segments(schedule):
    """Map method checkpoints to graph bodies, keeping historical segment names."""
    names = []
    for stage in schedule.stages:
        last = stage[-1]
        if last == 'J0':
            names.append('A')
        elif last == 'J1':
            names.append('B1r' if len(stage) == 2 else 'B1n')
        elif last == 'J2':
            names.append('B2')
        elif last == 'J3':
            names.append('C')
        else:
            names.append(last)
    return tuple(names)


CHAIN_SEGMENTS = ('A', 'B1r', 'B1n', 'B2', 'C', 'P')
FIXED_CONFIGS = tuple(str(Schedule(j, r)) for j in range(4) for r in range(j + 1))
CHAIN_VARIANTS = tuple((j, refresh, schedule_segments(Schedule(j, int(refresh))) + ('P',))
                       for j, refresh in ((3, True), (2, True), (2, False),
                                          (1, True), (1, False)))


def chain_variants(*, capacity=False):
    """Launch candidates in historical score tie order, including publication."""
    if capacity:
        return tuple(CHAIN_VARIANTS[i] for i in (0, 1, 3, 2, 4))
    return CHAIN_VARIANTS


def normalized_label(label):
    return isinstance(label, str) and re.fullmatch(r'J0d0|J[1-9][0-9]*', label) is not None


def completed_label(passes_run, *, max_depth=None):
    passes_run = int(passes_run)
    if passes_run < 1:
        raise ValueError('a served block must have at least one computed pass')
    depth = passes_run - 1
    if max_depth is not None:
        depth = min(depth, max_depth)
    return 'J0d0' if depth == 0 else f'J{depth}'


PASS_LABELS = tuple(completed_label(n) for n in range(1, 5))


def resolve_tau(table, label, *, bundle=None, recipe=None):
    """Dataset pooling is already resolved; this fallback pools pass labels only."""
    if not normalized_label(label):
        raise ValueError(f'invalid normalized pass label {label!r}')
    key = label if label in table else 'pooled'
    if key not in table:
        raise ValueError(f'missing tau for {label}; bundle={bundle}; recipe={recipe}; '
                         'no explicit label or pooled key')
    value = table[key]
    return float(value['value'] if isinstance(value, dict) else value)
