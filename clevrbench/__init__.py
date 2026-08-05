"""clevrbench — a modular harness for evaluating VLM compositional reasoning on CLEVR.

The pieces are deliberately separable:

    taxonomy   what makes a CLEVR question compositional (type, depth, hops)
    answers    mapping free-form model text onto CLEVR's 28-answer vocabulary
    data       locating/fetching CLEVR, whatever state the machine is in
    subset     carving a small, self-contained, reproducible eval set
    models     the adapter interface every model plugs into
    runner     the evaluation loop
    metrics    accuracy sliced along the compositional axes, with error bars
    report     console tables + results.json

Adding a model touches only `models/`. Nothing else in the harness knows
which model it is scoring.
"""

__version__ = "0.1.0"
