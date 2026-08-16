"""JSON rendering of a CampaignReportDocument: the complete, machine-
readable record. dataclasses.asdict() recurses through the entire
document tree for free -- every result type in validation/ and
campaign/correction.py is already a plain dataclass of floats/strings/
bools/nested dataclasses, so no hand-written serializer is needed.

No `default=` fallback on json.dumps: if some field ever turned out to
be non-JSON-native (e.g. a stray numpy scalar that skipped a float()
cast upstream), that should surface immediately as a loud error, not get
silently stringified into a JSON string where it would look like a
number but no longer be one.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from discovery_engine.reporting.schema import CampaignReportDocument


def to_json(document: CampaignReportDocument, indent: int = 2) -> str:
    return json.dumps(dataclasses.asdict(document), indent=indent)


def write_json(document: CampaignReportDocument, path: str | Path) -> None:
    Path(path).write_text(to_json(document), encoding="utf-8")
