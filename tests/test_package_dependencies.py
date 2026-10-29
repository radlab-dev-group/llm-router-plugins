"""
The package declares every dependency its modules import at module level.

A dependency that is only present in the development interpreter is not a
dependency: a fresh install of the package fails on the first import, and the
test suite that runs on it fails at collection — which is exactly how a
`pandas` import in the FastMasker core survived until CI ran on a clean
checkout.  Reading the metadata is cheap and needs no interpreter matrix, so
the check belongs with the tests rather than with one pipeline definition.

A module that imports something only an extra installs is legitimate; the
subtrees that do are named below, so an extra cannot quietly become a
requirement of the whole package.
"""

import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "llm_router_plugins"

# Subtrees whose heavy imports an extra provides, so a module-level import
# there is not a dependency of the bare package.
EXTRA_SCOPED = {
    "utils/rag": {
        "langchain",
        "langchain_community",
        "langchain_core",
        "faiss",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
        "numpy",
        "PIL",
    },
    "utils/routing": {"faiss", "sentence_transformers", "numpy"},
    "cli/plugins/utils/agentic_knowledge": {"redis", "sse_lib"},
}


def _module_level_imports(path):
    """Yield the top-level module names *path* imports when imported."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.module.split(".")[0]


def _setup_keyword(name):
    """Return the literal value of one ``setup(...)`` keyword of setup.py."""
    tree = ast.parse((ROOT / "setup.py").read_text(), filename="setup.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup":
            for keyword in node.keywords:
                if keyword.arg == name:
                    return ast.literal_eval(keyword.value)
    raise AssertionError(f"setup.py declares no {name}")


def _distribution_name(spec):
    """The package name of one requirement spec, markers and extras dropped."""
    spec = spec.split(";", 1)[0].strip()  # environment markers
    spec = spec.split("[", 1)[0].strip()  # extras of this requirement
    for marker in (">=", "==", "<=", "!=", "~=", ">", "<"):
        spec = spec.split(marker, 1)[0]
    return spec.strip()


@pytest.fixture(scope="module")
def declared():
    """Every package name the distribution can install."""
    required = _setup_keyword("install_requires")
    extras = _setup_keyword("extras_require")
    names = set()
    for spec in list(required) + [
        spec for group in extras.values() for spec in group
    ]:
        names.add(_distribution_name(spec))
    return {
        "required": {_distribution_name(spec) for spec in required},
        "all": names,
    }


def _top_level(name):
    """The distribution-style name of an import."""
    return name.replace("_", "-")


@pytest.mark.parametrize(
    "module", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT))
)
def test_a_module_imports_only_declared_dependencies(module, declared):
    """No module reaches for a package the distribution cannot install."""
    relative = module.relative_to(PACKAGE).as_posix()
    if relative == "utils/pipeline.py":
        pytest.skip("imports the internal, non-shipped llm_router_api")
    allowed = set()
    for prefix, provided in EXTRA_SCOPED.items():
        if relative.startswith(prefix + "/") or relative == prefix:
            allowed |= provided
    for imported in _module_level_imports(module):
        if imported in sys.stdlib_module_names or imported == PACKAGE.name:
            continue
        if imported in allowed:
            continue
        candidates = {imported, _top_level(imported)}
        assert candidates & declared["all"], (
            f"{relative} imports {imported!r} at module level, but no "
            "install_requires or extra declares it — a fresh install fails "
            "on import"
        )


@pytest.mark.parametrize("package", ["pandas", "requests", "numpy"])
def test_the_core_dependencies_are_not_an_extra(package, declared):
    """What the maskers and the plugin interface import is always installed.

    The FastMasker core builds a mapping DataFrame and the plugin interface is
    HTTP-based; both are imported by every built-in plugin, so neither can be
    satisfied by an optional extra.
    """
    assert package in declared["required"], (
        f"{package} must stay in install_requires: the core imports it at "
        "module level"
    )
