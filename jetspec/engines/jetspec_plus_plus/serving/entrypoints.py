"""Inspect GPU tool arguments on CPU without importing their execution runtime."""
import ast
from pathlib import Path
import sys


def cpu_help(source, *, arguments_source=None, function=None):
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
    shared = {}
    if arguments_source is not None:
        tree = ast.parse(Path(arguments_source).read_text())
        definitions = next(node.value for node in tree.body
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == '_COMMON'
                                   for target in node.targets))
        shared = {key.value: value for key, value in zip(definitions.keys, definitions.values)}
        tree = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == function)
    seen = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == '_argument'):
            flag, defaults = shared[node.args[1].value].elts
            keywords = {keyword.arg: keyword for keyword in defaults.keywords}
            keywords.update({keyword.arg: keyword for keyword in node.keywords})
            node = ast.Call(func=ast.Attribute(value=ast.Name(id='parser'), attr='add_argument'),
                            args=[flag], keywords=list(keywords.values()))
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
