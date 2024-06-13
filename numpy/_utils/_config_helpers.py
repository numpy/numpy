from enum import Enum


class ConfigDisplayModes(Enum):
    stdout = "stdout"
    dicts = "dicts"


def cleanup_empty_dict_values(d):
    """
    Removes empty values in a `dict` recursively
    This ensures we remove values that Meson could not provide to CONFIG
    """
    if isinstance(d, dict):
        return {
            k: cleanup_empty_dict_values(v)
            for k, v in d.items()
            if v and cleanup_empty_dict_values(v)
        }
    else:
        return d


def check_pyyaml():
    import yaml

    return yaml


def print_or_return_config(mode, config):
    if mode == ConfigDisplayModes.stdout.value:
        try:  # Non-standard library, check import
            yaml = check_pyyaml()

            print(yaml.dump(config))
        except ModuleNotFoundError:
            import warnings
            import json

            warnings.warn("Install `pyyaml` for better output", stacklevel=1)
            print(json.dumps(config, indent=2))
    elif mode == ConfigDisplayModes.dicts.value:
        return config
    else:
        raise AttributeError(
            "Invalid `mode`, use one of: "
            f"{', '.join([e.value for e in ConfigDisplayModes])}"
        )
