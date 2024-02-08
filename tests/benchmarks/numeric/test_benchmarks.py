from vanZadelhoff_1_1D import create_and_run_model as vanZadelhoff_1_setup_and_run

from magrittetorch.model.model import Model
import pytest


class TestNumeric:
    #incremental testing; see conftest.py or pytest.ini
    # @pytest.mark.incremental
    class TestVanZadelhoff1:
        #testing whether the results are correct
        def vanZadelhoff1a_setup_and_run(self):
            vanZadelhoff_1_setup_and_run(nosave=True, a_or_b='a')
