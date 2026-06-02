.PHONY: format
format:
	uvx pre-commit run -a

.PHONY: type
type:
	uvx ty check

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: test-unit
test-unit:
	uv run pytest -m unit

.PHONY: test-integration
test-integration:
	uv run pytest -m integration

.PHONY: test-gpu
test-gpu:
	uv run pytest -m gpu

.PHONY: test-hardware
test-hardware:
	uv run pytest -m hardware

.PHONY: build
build:
	uv build
