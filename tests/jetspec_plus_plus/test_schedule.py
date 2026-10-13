"""Admission and subsequent rounds share the evaluated fixed schedule."""
from golden_values import golden
import pytest

from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig


@pytest.mark.parametrize('schedule', ['J1-R1', 'J3-R3', 'J0-R0'])
def test_fixed_schedule(schedule):
    # Ported from the JetSpec++ reference implementation.
    config = JetSpecPPConfig(schedule=schedule)
    assert config.operations == golden('value_1')
    assert config.operations[0] == 'J0'
    assert config.rows_min == 8
    assert config.block_size == 16
