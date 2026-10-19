"""Published local paths exist and README Python commands expose usable help."""
import re
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('document', ['README.md', 'CONTRIBUTING.md'])
def test_documented_repo_paths_exist(document):
    text = (ROOT / document).read_text()
    code = '\n'.join(block or inline for block, inline in
                     re.findall(r'```[^\n]*\n(.*?)```|`([^`\n]+)`', text, re.S))
    paths = re.findall(r'(?<![\w/$.{])(jetspec/|bench/|tests/|examples/)([\w./-]*)', code)
    assert paths, f'no repository paths found in {document}'
    missing = [prefix + suffix for prefix, suffix in paths
               if not (ROOT / (prefix + suffix)).exists()]
    assert not missing, f'{document}: {missing}'


def test_readme_python_commands_have_help():
    scripts = set(re.findall(r'\bpython\s+((?:jetspec|bench|tests|examples)/[\w./-]+\.py)',
                             (ROOT / 'README.md').read_text()))
    assert scripts
    for script in sorted(scripts):
        result = subprocess.run([sys.executable, script, '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f'{script}: {result.stdout}\n{result.stderr}'
