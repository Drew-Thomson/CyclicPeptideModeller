# CyclicPeptideModeller

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/Drew-Thomson/LoopMover/actions/workflows/tests.yml/badge.svg)](https://github.com/Drew-Thomson/LoopMover/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Implementation of kinematic chain analysis for loops in protein structures, applied to the design of macrocyclic peptides.

The underlying kinematic closure algorithm is detailed in [10.1002/jcc.10416](https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.10416).

## Overview

Provides tools to build, close, and optimise cyclic peptide structures using kinematic closure and OpenMM for energy minimisation.

## Demonstration

See `demo.ipynb` for a working example of initialising and optimising a target cyclic peptide sequence.

```python
from loopmover.optimiser import CyclicPeptideOptimiser

sequence = 'rQpqRePQ'
optimiser = CyclicPeptideOptimiser(sequence)
optimiser.build_start_mac()
optimiser.amber_setup()
optimiser.optimise(n_iter=5, samplesize=20, max_iter=5)
```
