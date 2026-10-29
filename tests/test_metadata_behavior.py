import pytest

import pyvista as pv

from kneemesher import metadata as m


@pytest.fixture(autouse=True)
def disable_plot(monkeypatch):
    """Prevent PyVista from opening interactive windows during tests."""
    monkeypatch.setattr(pv.PolyData, "plot", lambda self, *a, **k: None)
    monkeypatch.setattr(pv.MultiBlock, "plot", lambda self, *a, **k: None)
    yield


def test_single_container_proxy_and_property():
    @m.with_metadata(subject=m.NonBlank())
    class C:
        def __init__(self, **kwargs):
            # accept dynamic kwargs forwarded by the decorator
            pass

    inst = C(subject="abc")

    # public property
    assert inst.metadata.subject == "abc"

    # proxied attribute access (default proxy=True)
    assert inst.subject == "abc"

    # dir includes metadata fields
    assert "subject" in dir(inst)


def test_repeated_decorations_and_ambiguity():
    # decorate the class twice with distinct container names
    @m.with_metadata(container_name="kmeta", subject=m.NonBlank())
    @m.with_metadata(container_name="pv_metadata", subject=m.NonBlank())
    class T:
        def __init__(self, **kwargs):
            pass

    t = T()

    # create two containers and attach them to the private mangled names
    kmeta = m.MetaContainer(subject="kval")
    pvmeta = m.MetaContainer(subject="pvval")

    # decorator uses private attribute names like "__kmeta" and "__pv_metadata"
    setattr(t, "__kmeta", kmeta)
    setattr(t, "__pv_metadata", pvmeta)

    # explicit access via properties
    assert t.kmeta.subject == "kval"
    assert t.pv_metadata.subject == "pvval"

    # ambiguous flat attribute should raise with helpful message
    with pytest.raises(AttributeError) as exc:
        _ = t.subject
    msg = str(exc.value)
    assert "Ambiguous metadata attribute" in msg
    assert "kmeta" in msg and "pv_metadata" in msg


def test_validators():
    nb = m.NonBlank()
    assert nb.validate("name", "abc") == "abc"
    with pytest.raises(ValueError):
        nb.validate("name", "   ")

    tp = m.Timepoint()
    assert tp.validate("t", 3) == 3
    from datetime import datetime

    dt = datetime.now()
    assert isinstance(tp.validate("t", dt), (int, float))

    re = m.RegexMatched(r"^\d+$")
    assert re.validate("n", "123") == "123"
    with pytest.raises(ValueError):
        re.validate("n", "abc")

    ve = m.ValidEnum(m.Side)
    assert ve.validate("side", "left") == m.Side.LEFT

    vvs = m.ValidVersionString()
    assert vvs.validate("v", "1.2.3") == "1.2.3"
