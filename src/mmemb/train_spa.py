"""Train Stage I with the SPA pairwise objective."""

import json

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    config = load_config("spa", argv)
    if config["dry_run"]:
        print(json.dumps(config, indent=2))
        return
    from .trainer import run_training

    run_training(config)


if __name__ == "__main__":
    main()
