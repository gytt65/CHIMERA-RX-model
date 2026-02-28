"""Non-regression guard: CHIMERA-RX imports must not alter OMEGA class behavior."""

from omega_features import get_features
from omega_model import OMEGAModel


def test_omega_class_unchanged_after_chimera_imports():
    before_method = OMEGAModel.price_option
    before_flags = get_features().to_dict().copy()

    import chimera_rx_engine  # noqa: F401
    import chimera_rx_data_adapter  # noqa: F401
    import chimera_rx_scoring  # noqa: F401
    import chimera_rx_tab  # noqa: F401

    after_method = OMEGAModel.price_option
    after_flags = get_features().to_dict().copy()

    assert after_method is before_method
    assert after_flags == before_flags
