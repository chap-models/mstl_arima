"""``python -m chap_mstl_arima`` entry point.

Kept as a module entry point (rather than a console script) because the
chapkit ``ShellModelRunner`` copies the project into a scratch workspace and
runs commands there with ``cwd`` on ``sys.path`` - ``python -m`` resolves
without the package having to be installed into the workspace.
"""

from chap_mstl_arima.cli import main

if __name__ == "__main__":
    main()
