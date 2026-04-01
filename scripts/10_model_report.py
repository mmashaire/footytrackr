from __future__ import annotations

import json

from footytrackr.reporting import write_model_report


def main() -> None:
    report, json_path, markdown_path = write_model_report()

    print("✅ Model report complete.")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved Markdown report to {markdown_path}")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
