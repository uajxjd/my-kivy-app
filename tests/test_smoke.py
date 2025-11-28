from fc3d.data import load_csv
from pathlib import Path

def test_load():
    csv = Path(__file__).parent.parent / "UAFC3D.csv"
    df = load_csv(csv)
    assert not df.empty