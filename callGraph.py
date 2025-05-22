from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from fraud_analysis_agent import main

with PyCallGraph(output=GraphvizOutput()):
    main()
