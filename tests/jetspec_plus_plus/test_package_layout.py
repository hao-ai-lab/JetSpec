"""The JetSpec++ method package imports without loading model code."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('path', ['jetspec.methods.jetspec_plus_plus'])
def test_import_stays_model_free(path):
    script = f'''
import sys
import {path}
for name in sys.modules:
    assert not name.startswith(('transformers', 'jetspec.models', 'jetspec.core',
                               'jetspec.inference_engine',
                               'jetspec.methods.jetspec.draft',
                               'jetspec.methods.jetspec_plus_plus.models')), name
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
