# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re
from typing import List

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("src/requirements.txt", "r", encoding="utf-8") as f:
    requirements: List[str] = f.read().splitlines()

with open("src/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in sim/__init__.py"
version: str = version_re.group(1)


setup(
    name="depths",
    version=version,
    description="depths experiments",
    author="Paweł Budzianowski",
    url="https://github.com/kscalelabs/depths",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # python_requires=">=3.8, <3.9",
    install_requires=requirements,
    tests_require=requirements,
)
