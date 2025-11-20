"""Build T0 events from Orange Book."""
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.label.orange_book_parser import OrangeBookParser


def main():
    """Build events.parquet from Orange Book data."""
    config = load_config()
    parser = OrangeBookParser(config)
    
    print("Computing T0 events from Orange Book...")
    events = parser.compute_t0_events()
    
    print(f"Found {len(events)} events")
    
    # Save to artifacts
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = artifacts_dir / "events.parquet"
    events.to_parquet(output_path, index=False)
    print(f"Saved events to {output_path}")
    
    # Also save a sample CSV for inspection
    events.head(100).to_csv(artifacts_dir / "events_sample.csv", index=False)
    print(f"Saved sample to {artifacts_dir / 'events_sample.csv'}")


if __name__ == "__main__":
    main()

