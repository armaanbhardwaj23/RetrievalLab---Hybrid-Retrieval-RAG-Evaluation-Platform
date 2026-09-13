"""Allow ``python -m retrievallab`` as an alternative to the CLI script."""

from retrievallab.cli import main


raise SystemExit(main())
