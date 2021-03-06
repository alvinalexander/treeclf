#!/usr/bin/env bash
# Script that provides the right context for import statements.
# For instance, if your current working directory is ./treeclf
# you can execute a script like this: ./treeclf-py.bash core/tests/test_graph.py
# Note that all import statements must follow the following format:
# import toplevel_package.subpackage.module
# e.g from core.graph.digraph import DirectedGraph

PYTHONPATH=. python $*