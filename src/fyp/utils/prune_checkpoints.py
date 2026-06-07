import torch
from pathlib import Path
import sys

def prune_checkpoints(models_dir: Path, target_runs=None):
    if target_runs is None:
        target_runs = ["run_1", "run_2", "run_3", "run_4", "run_5"]

    print(f"Scanning directory: {models_dir} for target runs: {target_runs}")
    
    pth_files = []
    for run in target_runs:
        run_dir = models_dir / run
        if run_dir.exists():
            pth_files.extend(run_dir.glob("**/*.pth"))
            
    # Also search directly in models_dir if no subdirectories are matched
    if not pth_files and not any((models_dir / run).exists() for run in target_runs):
        pth_files = list(models_dir.glob("*.pth"))

    print(f"Found {len(pth_files)} checkpoint files to prune.")
    total_bytes_saved = 0

    for path in pth_files:
        try:
            original_size = path.stat().st_size
            # Load the checkpoint state dict on CPU
            state_dict = torch.load(path, map_location="cpu")
            
            # Keep only the decoder head parameters for these old frozen runs
            pruned_state_dict = {k: v for k, v in state_dict.items() if "head" in k}
            
            if len(pruned_state_dict) == len(state_dict):
                print(f"Skipping {path.name} (already pruned or only contains head)")
                continue

            # Save the pruned state dict back to the file
            torch.save(pruned_state_dict, path)
            new_size = path.stat().st_size
            
            saved_mb = (original_size - new_size) / (1024 * 1024)
            total_bytes_saved += (original_size - new_size)
            print(f"Pruned {path.relative_to(models_dir)}: {original_size/(1024*1024):.1f}MB -> {new_size/(1024*1024):.1f}MB (Saved {saved_mb:.1f}MB)")
            
        except Exception as e:
            print(f"Failed to prune {path}: {e}")

    print(f"\nCompleted! Total space reclaimed: {total_bytes_saved / (1024 * 1024 * 1024):.2f} GB")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    models_dir = project_root / "models"
    
    # Allow passing custom run subdirectories via arguments if needed
    args = sys.argv[1:]
    target_runs = args if args else None
    
    prune_checkpoints(models_dir, target_runs)
