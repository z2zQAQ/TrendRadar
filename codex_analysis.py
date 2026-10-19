"""Offline entry point for the manual Codex analysis workflow."""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from trendradar.commands.codex_analysis import main


if __name__ == "__main__":
    main()
