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



parse_schedule = Schedule.parse
