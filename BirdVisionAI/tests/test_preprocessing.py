import sys
from pathlib import Path

# Add project root and app directory to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "app"))
sys.path.insert(1, str(root))

from preprocessing.transforms import get_transform

def test_preprocessing():
    transform = get_transform()
    assert transform is not None
