import sys
from pathlib import Path

# make `import src...` work when pytest is launched from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
