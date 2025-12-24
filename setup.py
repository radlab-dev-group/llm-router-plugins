import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
VERSION = (HERE / ".version").read_text(encoding="utf-8")

setup(
    name="llm-router-plugins",
    version=VERSION,
    description="Plugins for the LLM Router (guardrails, maskers, etc.)",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="RadLab.dev Team",
    url="https://github.com/radlab-dev-group/llm-router-plugins",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
    entry_points={
        "console_scripts": [
            "llm-router-rag-langchain=cli.plugins.rag.langchain:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
