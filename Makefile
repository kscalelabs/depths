# Makefile

define HELP_MESSAGE
nap


endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all


# ------------------------ #
#          Build           #
# ------------------------ #

install:
	@pip install --verbose -e .
.PHONY: install

install-dev:
	@pip install --verbose -e '.[dev]'
.PHONY: install

install-third-party:
	@git submodule update --init --recursive

build-ext:
	@python setup.py build_ext --inplace
.PHONY: build-ext

clean:
	rm -rf build dist *.so **/*.so **/*.pyi **/*.pyc **/*.pyd **/*.pyo **/__pycache__ *.egg-info .eggs/ .ruff_cache/
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #

format:
	@isort --profile black src
	@black src
	@ruff format src
.PHONY: format

format-cpp:
	@clang-format -i $(shell find . -name '*.cpp' -o -name '*.h')
	@cmake-format -i $(shell find . -name 'CMakeLists.txt' -o -name '*.cmake')
.PHONY: format-cpp

static-checks:
	@isort --check-only src
	@black --diff --check src
	@ruff check src
.PHONY: lint

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test
