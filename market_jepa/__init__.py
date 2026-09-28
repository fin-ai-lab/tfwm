"""market-jepa: Time Series JEPA for market data."""

from market_jepa.tempfiles import configure_tempdir

# This machine's system /tmp is intentionally small. Establish the safe
# default before libraries such as Mosaic Streaming first consult tempfile.
configure_tempdir()

__all__ = ["configure_tempdir"]
