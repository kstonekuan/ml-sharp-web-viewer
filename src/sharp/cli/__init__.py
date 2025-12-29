"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import cloud, predict, render


@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")
main_cli.add_command(render.render_cli, "render")
main_cli.add_command(cloud.cloud_cli, "cloud")
