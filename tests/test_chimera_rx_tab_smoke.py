from chimera_rx_tab import render


class _DummyCol:
    def metric(self, *args, **kwargs):
        return None


class _DummySt:
    def __init__(self):
        self.outputs = []

    def markdown(self, *args, **kwargs):
        self.outputs.append(("markdown", args, kwargs))

    def caption(self, *args, **kwargs):
        self.outputs.append(("caption", args, kwargs))

    def info(self, *args, **kwargs):
        self.outputs.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.outputs.append(("warning", args, kwargs))

    def columns(self, n):
        return [_DummyCol() for _ in range(n)]

    def dataframe(self, *args, **kwargs):
        self.outputs.append(("dataframe", args, kwargs))

    def json(self, *args, **kwargs):
        self.outputs.append(("json", args, kwargs))


def test_chimera_rx_tab_smoke_renders_without_crashing():
    st = _DummySt()
    session_state = {
        "spot": 22500.0,
        "selected_strike": 22600.0,
        "T": 0.07,
        "option_type": "CE",
        "market_price": 185.0,
        "india_vix": 14.0,
        "pcr_oi": 1.0,
    }

    render(st, session_state)

    assert "chimera_rx_latest" in session_state
    assert "chimera_rx_ptd_timeline" in session_state
    assert "chimera_rx_ccas_history" in session_state
    assert any(kind == "json" for kind, *_ in st.outputs)
