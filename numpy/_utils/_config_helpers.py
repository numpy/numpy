from enum import Enum


class _ConfigDisplayModes(Enum):
    stdout = "stdout"
    dicts = "dicts"


def _cleanup_empty_dict_values(d):
    """
    Removes empty values in a `dict` recursively
    This ensures we remove values that Meson could not provide to CONFIG
    """
    if isinstance(d, dict):
        return {
            k: _cleanup_empty_dict_values(v)
            for k, v in d.items()
            if v and _cleanup_empty_dict_values(v)
        }
    else:
        return d


def _check_pyyaml():
    import yaml

    return yaml


def _print_or_return_config(mode, config):
    if mode == _ConfigDisplayModes.stdout.value:
        try:  # Non-standard library, check import
            yaml = _check_pyyaml()

            print(yaml.dump(config))
        except ModuleNotFoundError:
            import json
            import warnings

            warnings.warn("Install `pyyaml` for better output", stacklevel=1)
            print(json.dumps(config, indent=2))
    elif mode == _ConfigDisplayModes.dicts.value:
        return config
    else:
        raise AttributeError(
            "Invalid `mode`, use one of: "
            f"{', '.join([e.value for e in _ConfigDisplayModes])}"
        )
