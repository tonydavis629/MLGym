"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Utility functions for MLGym.

"""

from __future__ import annotations


def multiline_representer(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

def get_devices() -> list[int]:
    """Get the list of available GPUs"""
    try:
        import GPUtil
    except ImportError:
        msg = "GPUtil is not installed. Please install it with `pip install gputil`."
        raise ImportError(msg)
    # get all the GPUs
    gpus = GPUtil.getAvailable(limit=100)
    return gpus
