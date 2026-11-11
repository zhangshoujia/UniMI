"""Continue the SPA adapter with the CCA objective."""

import json

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    config = load_config("cca", argv)
    if config["dry_run"]:
        print(json.dumps(config, indent=2))
        return
    from .trainer import run_training

    run_training(config)


if __name__ == "__main__":
    main()
