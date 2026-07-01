import numpy as np
import pandas as pd

from pdmuse import ChoiceDataset


def test_from_long_dataframe_round_trips_choice_arrays():
    frame = pd.DataFrame(
        {
            "case": [1, 1, 2, 2],
            "alt": ["car", "bus", "car", "bus"],
            "chosen": [1, 0, 0, 1],
            "time": [3.0, 5.0, 4.0, 2.0],
            "cost": [8.0, 2.0, 7.0, 3.0],
        }
    )

    data = ChoiceDataset.from_long_dataframe(
        frame,
        choice_id_col="case",
        alternative_col="alt",
        choice_col="chosen",
        feature_cols=["time", "cost"],
    )

    assert data.X.shape == (2, 2, 2)
    assert data.y.tolist() == [0, 1]
    assert data.alt_names == ("car", "bus")
    assert np.allclose(data.to_long_dataframe()["chosen"].to_numpy(), frame["chosen"].to_numpy())
