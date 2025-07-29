#!/usr/bin/env python3
"""
Main CLI entry point for the refactored playlista application.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.presentation.cli.cli_interface import CLIInterface


def main():
    """Main CLI entry point."""
    import sys
    from src.presentation.cli.cli_interface import CLIInterface
    args = sys.argv[1:]
    if args and args[0] in {"help", "--help", "-h"}:
        CLIInterface()._show_help()
        return 0
    try:
        cli = CLIInterface()
        return cli.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 