"""CohortData: the container every discovery method reads from.

Deliberately dumb (a dict of DataFrames) so that "which modality did this
method touch" is just "which keys did it look up", not something a method
has to self-report honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CohortData:
    """modality name -> DataFrame indexed by sample id, columns = features.

    `covariates` (optional) is a DataFrame indexed by sample id, columns =
    clinical covariates (grade, stage, ...), used by the validation gate's
    covariate-independence checks. It is deliberately not a "modality" so
    discovery methods that only ever look up `data.modalities[...]` cannot
    accidentally read it.
    """

    modalities: dict[str, pd.DataFrame]
    covariates: pd.DataFrame | None = field(default=None)

    @property
    def sample_ids(self) -> list[str]:
        ids: set[str] = set()
        for df in self.modalities.values():
            ids.update(df.index)
        return sorted(ids)
