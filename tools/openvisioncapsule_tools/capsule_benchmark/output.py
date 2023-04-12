from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd


def generate_output(output: pd.DataFrame, *,
                    csv_path: Optional[Path] = None,
                    graph_path: Optional[Path] = None):
    if csv_path is not None:
        generate_csv_output(output, csv_path)
    if graph_path is not None:
        generate_graph_output(output, graph_path)


def generate_csv_output(output: pd.DataFrame, csv_path: Path):
    output.to_csv(csv_path)


def generate_graph_output(output: pd.DataFrame, graph_path: Path):
    num_workers = alt.X('num_workers:O',
                        axis=alt.Axis(title="Parallelization"))
    fps = alt.Y('fps:Q', axis=alt.Axis(title="FPS"))
    capsules = alt.Color('capsule_name:N', legend=alt.Legend(title="Capsule"))

    graph: alt.Chart = alt.Chart(output).mark_line().encode(
        x=num_workers,
        y=fps,
        color=capsules,
    )

    graph: alt.Chart = graph.properties(
        width=600,
        height=600
    )

    graph.save(str(graph_path))
