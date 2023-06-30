test:
	pytest .


lint:
	@echo
	ruff viper
	@echo
	blue --check --diff --color viper
	@echo
	mypy viper
	@echo
	pip-audit


format:
	ruff --silent --exit-zero --fix viper
	blue viper