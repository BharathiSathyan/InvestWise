import inspect

from risklens import data, metrics, optimize, costs, backtest

modules = {
    "DATA": data,
    "METRICS": metrics,
    "OPTIMIZE": optimize,
    "COSTS": costs,
    "BACKTEST": backtest
}

for module_name, module in modules.items():
    print(f"\n{'=' * 50}")
    print(module_name)
    print("=" * 50)

    for name, function in inspect.getmembers(
        module, inspect.isfunction
    ):
        if not name.startswith("_"):
            print(f"\n{name}{inspect.signature(function)}")
            print(inspect.getdoc(function) or "No docstring")