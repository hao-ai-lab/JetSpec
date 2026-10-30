"""Inspect GPU tool arguments on CPU without importing their execution runtime."""
import ast
from pathlib import Path
import sys


def cpu_help(source):
    """Print source-declared CLI options only when help was requested.

    No parser defaults or callbacks are evaluated. The original parser still
    handles execution, including validation and dynamically supplied defaults.
    """
    if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        return
    path = Path(source)
    tree = ast.parse(path.read_text())
    print(f"usage: {path.name} [arguments]\n")
    print(ast.get_docstring(tree) or f"Inspect or execute {path.stem}.")
    print("\nCPU help: source-declared arguments; execution uses the original parser.")
    print("  -h, --help  Show this help without loading runtime dependencies.")
    seen = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("add_argument", "add_parser")):
            continue
        names = [arg.value for arg in node.args
                 if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        if not names:
            continue
        label = ", ".join(names)
        details = []
        for keyword in node.keywords:
            if keyword.arg in ("help", "choices", "default", "required", "nargs"):
                if isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                else:
                    value = ast.unparse(keyword.value)
                details.append(str(value) if keyword.arg == "help"
                               else f"{keyword.arg}={value}")
        line = f"  {label}  {'; '.join(details)}"
        if line not in seen:
            print(line)
            seen.add(line)
    raise SystemExit(0)


def forward(source, namespace):
    """Execute an immutable upstream tool with its original file location."""
    path = Path(source)
    if namespace.get("__name__") == "__main__":
        cpu_help(path)
    sys.path.insert(0, str(path.parent))
    namespace["__file__"] = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
