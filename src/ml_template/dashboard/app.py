"""Six-tab read-only modelling dashboard."""

from __future__ import annotations

import argparse

from dash import Dash, Input, Output, dcc, html
from sqlalchemy import Engine

from ml_template.dashboard.pages import (
    assets_page,
    calibration_page,
    compare_page,
    experiment_page,
    run_page,
    sweeps_page,
)


def create_app(engine: Engine | None = None) -> Dash:
    if engine is None:
        from ml_template.db.connection import get_engine

        engine = get_engine()
    app = Dash(__name__, title="Model experiments")

    def layout():
        return html.Div(
            [
                dcc.Tabs(
                    id="dashboard-tab",
                    value="run",
                    children=[
                        dcc.Tab(label="Assets", value="assets"),
                        dcc.Tab(label="Sweeps", value="sweeps"),
                        dcc.Tab(label="Experiments", value="experiments"),
                        dcc.Tab(label="Single run", value="run"),
                        dcc.Tab(label="Compare", value="compare"),
                        dcc.Tab(label="Calibration", value="calibration"),
                    ],
                ),
                html.Div(run_page.layout(engine), id="single-run-container"),
                html.Div(
                    experiment_page.layout(engine),
                    id="experiments-container",
                    style={"display": "none"},
                ),
                html.Div(
                    assets_page.layout(engine), id="assets-container", style={"display": "none"}
                ),
                html.Div(
                    sweeps_page.layout(engine), id="sweeps-container", style={"display": "none"}
                ),
                html.Div(
                    compare_page.layout(engine), id="compare-container", style={"display": "none"}
                ),
                html.Div(
                    calibration_page.layout(engine),
                    id="calibration-container",
                    style={"display": "none"},
                ),
            ]
        )

    app.layout = layout
    for page in (
        run_page,
        experiment_page,
        assets_page,
        sweeps_page,
        compare_page,
        calibration_page,
    ):
        page.register_callbacks(app, engine)

    @app.callback(
        Output("single-run-container", "style"),
        Output("experiments-container", "style"),
        Output("assets-container", "style"),
        Output("sweeps-container", "style"),
        Output("compare-container", "style"),
        Output("calibration-container", "style"),
        Input("dashboard-tab", "value"),
    )
    def _tab(value):
        return tuple(
            {} if value == name else {"display": "none"}
            for name in ("run", "experiments", "assets", "sweeps", "compare", "calibration")
        )

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the experiment dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    create_app().run(port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
