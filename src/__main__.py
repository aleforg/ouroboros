"""Allows ``python -m ouroboros <args>`` invocation.

Used by the web dashboard to launch ``ouroboros run`` as a subprocess using the
same Python interpreter, avoiding PATH-resolution issues.
"""
from ouroboros.cli import main

main()
