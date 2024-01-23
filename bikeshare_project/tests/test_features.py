
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeathersitImputer


def test_age_variable_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variable=config.model_config.weathersit_var,  # wetahersit
    )
    assert np.isnan(sample_input_data.loc[12230,'weathersit'])

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject.loc[12230,'weathersit'] == 'Clear'