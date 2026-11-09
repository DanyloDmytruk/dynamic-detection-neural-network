"""Compatibility entry point for the recorded duration + event_count ablation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation import main
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--remove", "execution_duration_seconds", "event_count", "--name", "ablation_event_count"]
    main()
