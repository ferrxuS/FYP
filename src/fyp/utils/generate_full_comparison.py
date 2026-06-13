import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir: Path, run_dirs: list) -> dict:
    """Load all cl_results_*.json files from specified run directories or the main results dir"""
    strategy_results = {}
    
    valid_run_dirs = []
    if run_dirs:
        for r in run_dirs:
            if (results_dir / r).exists():
                valid_run_dirs.append(r)
                
    if not valid_run_dirs:
        detected = [p.name for p in results_dir.glob("run_*") if p.is_dir()]
        if detected:
            print(f"Auto-detected run directories: {detected}")
            valid_run_dirs = detected
            
    if valid_run_dirs:
        for run_dir_name in valid_run_dirs:
            run_path = results_dir / run_dir_name
            json_files = list(run_path.glob("cl_results_*.json"))
            print(f"Found {len(json_files)} results files in {run_dir_name}")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    strategy_key = json_file.stem.replace("cl_results_", "")
                    data['run_dir'] = run_dir_name
                    
                    if strategy_key in strategy_results:
                        existing_run = strategy_results[strategy_key]['run_dir']
                        strategy_results[f"{strategy_key}_{existing_run}"] = strategy_results.pop(strategy_key)
                        strategy_results[f"{strategy_key}_{run_dir_name}"] = data
                    else:
                        strategy_results[strategy_key] = data
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    else:
        json_files = list(results_dir.glob("cl_results_*.json"))
        if json_files:
            print(f"No run directories found. Loading {len(json_files)} files directly from {results_dir.name}")
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    strategy_key = json_file.stem.replace("cl_results_", "")
                    data['run_dir'] = "Direct"
                    strategy_results[strategy_key] = data
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    
    return strategy_results

def get_run_type(run_name: str, project_root: Path) -> str:
    models_dir = project_root / "models" / run_name
    if not models_dir.exists():
        try:
            run_num = int(run_name.replace("run_", ""))
            return "Unfrozen" if run_num >= 7 else "Frozen"
        except ValueError:
            return "Frozen"
            
    pth_files = list(models_dir.glob("*.pth"))
    if not pth_files:
        return "Frozen"
        
    try:
        import torch
        state_dict = torch.load(pth_files[0], map_location="cpu")
        has_backbone = any("backbone" in k for k in state_dict.keys())
        return "Unfrozen" if has_backbone else "Frozen"
    except Exception:
        size_mb = pth_files[0].stat().st_size / (1024 * 1024)
        return "Unfrozen" if size_mb > 50.0 else "Frozen"

def plot_combined_comparison(strategy_results: dict, save_path: Path, project_root: Path):
    """Plot accuracy, forgetting, and backward transfer comparison side-by-side"""
    import numpy as np
    
    # 1. Group the results by strategy name and backbone type
    grouped_results = {}
    has_frozen = False
    has_unfrozen = False
    
    for key, data in strategy_results.items():
        if data is None or 'metrics' not in data:
            continue
            
        # Extract clean strategy name
        if "_" in key and ("run_" in key or "run" in key):
            strat_name = key.rsplit("_", 1)[0]
        else:
            strat_name = key
            
        strat_name = strat_name.lower()
        run_name = data.get('run_dir', 'Direct')
        
        # Determine run type
        rtype = get_run_type(run_name, project_root)
        if rtype == "Frozen":
            has_frozen = True
        else:
            has_unfrozen = True
            
        if strat_name not in grouped_results:
            grouped_results[strat_name] = {}
        grouped_results[strat_name][rtype] = data

    # 2. Check if we should do side-by-side (both frozen and unfrozen are present)
    use_side_by_side = has_frozen and has_unfrozen

    if not grouped_results:
        print("No strategy results found. Comparison plot aborted.")
        return

    strategy_display_names = {
        'naive': 'Naive',
        'er': 'ER',
        'der++': 'DER++',
        'mer': 'MER',
        'si': 'SI',
        'lwf': 'LwF',
        'slca': 'SLCA',
        'er_ace': 'ER-ACE'
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    if use_side_by_side:
        # Grouped side-by-side plotting
        strategies = sorted(list(grouped_results.keys()))
        labels = [strategy_display_names.get(s, s.upper()) for s in strategies]
        num_strategies = len(strategies)
        x = np.arange(num_strategies)
        width = 0.35

        frozen_accs, unfrozen_accs = [], []
        frozen_forget, unfrozen_forget = [], []
        frozen_bwt, unfrozen_bwt = [], []

        for s in strategies:
            # Frozen
            f_data = grouped_results[s].get("Frozen")
            if f_data:
                frozen_accs.append(f_data['metrics']['Average Accuracy'])
                frozen_forget.append(f_data['metrics']['Forgetting'])
                frozen_bwt.append(f_data['metrics'].get('Backward Transfer', 0.0))
            else:
                frozen_accs.append(0.0)
                frozen_forget.append(0.0)
                frozen_bwt.append(0.0)

            # Unfrozen
            u_data = grouped_results[s].get("Unfrozen")
            if u_data:
                unfrozen_accs.append(u_data['metrics']['Average Accuracy'])
                unfrozen_forget.append(u_data['metrics']['Forgetting'])
                unfrozen_bwt.append(u_data['metrics'].get('Backward Transfer', 0.0))
            else:
                unfrozen_accs.append(0.0)
                unfrozen_forget.append(0.0)
                unfrozen_bwt.append(0.0)

        # Plot Average Accuracy
        axes[0].bar(x - width/2, frozen_accs, width, label='Frozen', color='#3B82F6', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[0].bar(x + width/2, unfrozen_accs, width, label='Unfrozen', color='#EF4444', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[0].set_ylabel('Average IoU / Accuracy', fontsize=12)
        axes[0].set_title('Average Accuracy\n(Higher is Better)', fontsize=12, fontweight='bold', pad=10)
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        axes[0].legend()

        # Plot Forgetting
        axes[1].bar(x - width/2, frozen_forget, width, label='Frozen', color='#3B82F6', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[1].bar(x + width/2, unfrozen_forget, width, label='Unfrozen', color='#EF4444', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[1].set_ylabel('Forgetting Rate', fontsize=12)
        axes[1].set_title('Forgetting\n(Lower is Better)', fontsize=12, fontweight='bold', pad=10)
        max_forget = max(max(frozen_forget), max(unfrozen_forget)) if frozen_forget or unfrozen_forget else 0.1
        axes[1].set_ylim([0, max(0.1, max_forget * 1.2)])
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        axes[1].legend()

        # Plot Backward Transfer (BWT)
        axes[2].bar(x - width/2, frozen_bwt, width, label='Frozen', color='#3B82F6', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[2].bar(x + width/2, unfrozen_bwt, width, label='Unfrozen', color='#EF4444', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[2].set_ylabel('Backward Transfer (BWT)', fontsize=12)
        axes[2].set_title('Backward Transfer\n(Higher is Better)', fontsize=12, fontweight='bold', pad=10)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1.0)
        min_bwt = min(min(frozen_bwt), min(unfrozen_bwt)) if frozen_bwt or unfrozen_bwt else -0.1
        max_bwt = max(max(frozen_bwt), max(unfrozen_bwt)) if frozen_bwt or unfrozen_bwt else 0.1
        axes[2].set_ylim([min(-0.1, min_bwt * 1.2), max(0.1, max_bwt * 1.2)])
        axes[2].grid(axis='y', linestyle='--', alpha=0.5)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        axes[2].legend()

    else:
        # Original single-bar plotting code
        strategies = []
        avg_accs = []
        forgettings = []
        bwts = []
        colors_bwt = []

        for name in sorted(strategy_results.keys()):
            results = strategy_results[name]
            if "_" in name and ("run_" in name or "run" in name):
                base_name, run_part = name.rsplit("_", 1)
                label = f"{strategy_display_names.get(base_name.lower(), base_name.upper())} ({run_part.replace('run', 'Run ')})"
            else:
                label = strategy_display_names.get(name.lower(), name.upper())
                
            strategies.append(label)
            avg_accs.append(results['metrics']['Average Accuracy'])
            forgettings.append(results['metrics']['Forgetting'])
            
            bwt = results['metrics'].get('Backward Transfer', 0.0)
            bwts.append(bwt)
            colors_bwt.append('green' if bwt >= 0 else 'red')

        # Plot Average Accuracy
        axes[0].bar(strategies, avg_accs, color='steelblue', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[0].set_ylabel('Average IoU / Accuracy', fontsize=12)
        axes[0].set_title('Average Accuracy\n(Higher is Better)', fontsize=12, fontweight='bold', pad=10)
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)

        # Plot Forgetting
        axes[1].bar(strategies, forgettings, color='coral', alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[1].set_ylabel('Forgetting Rate', fontsize=12)
        axes[1].set_title('Forgetting\n(Lower is Better)', fontsize=12, fontweight='bold', pad=10)
        max_forget = max(forgettings) if forgettings else 0.1
        axes[1].set_ylim([0, max(0.1, max_forget * 1.2)])
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)

        # Plot Backward Transfer (BWT)
        axes[2].bar(strategies, bwts, color=colors_bwt, alpha=0.85, edgecolor='black', linewidth=0.7)
        axes[2].set_ylabel('Backward Transfer (BWT)', fontsize=12)
        axes[2].set_title('Backward Transfer\n(Higher is Better)', fontsize=12, fontweight='bold', pad=10)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1.0)
        min_bwt, max_bwt = min(bwts) if bwts else -0.1, max(bwts) if bwts else 0.1
        axes[2].set_ylim([min(-0.1, min_bwt * 1.2), max(0.1, max_bwt * 1.2)])
        axes[2].grid(axis='y', linestyle='--', alpha=0.5)

        for ax in axes:
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=10)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Comparative Analysis of Continual Learning Strategies", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully generated comparison plot: {save_path}")

    print("\n" + "="*60)
    print("COMPARATIVE SUMMARY TABLE")
    print("="*60)
    print(f"| {'Strategy':<25} | {'Source Run':<12} | {'Backbone':<10} | {'Avg Accuracy':<12} | {'Forgetting':<10} | {'BWT':<10} |")
    print(f"|{'-'*27}|{'-'*14}|{'-'*12}|{'-'*14}|{'-'*12}|{'-'*12}|")
    for name, data in sorted(strategy_results.items()):
        metrics = data['metrics']
        avg_acc = metrics['Average Accuracy']
        forgetting = metrics['Forgetting']
        bwt = metrics.get('Backward Transfer', 0.0)
        run_name = data['run_dir']
        backbone_type = get_run_type(run_name, project_root)
        
        # Clean display name
        if "_" in name and ("run_" in name or "run" in name):
            base_name = name.rsplit("_", 1)[0]
        else:
            base_name = name
        disp_name = strategy_display_names.get(base_name.lower(), base_name.upper())
        
        print(f"| {disp_name:<25} | {run_name:<12} | {backbone_type:<10} | {avg_acc:<12.4f} | {forgetting:<10.4f} | {bwt:<10.4f} |")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine CL results from multiple runs into a single comparison chart.")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to main results directory")
    parser.add_argument("--runs", nargs="+", default=["run_2", "run_3"], help="Run subdirectories to load results from (space separated)")
    parser.add_argument("--output-name", type=str, default="cl_comparison.png", help="Filename of the saved comparison chart")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    results_dir_path = project_root / args.results_dir
    output_image_path = results_dir_path / args.output_name
    
    print(f"Project root: {project_root}")
    print(f"Loading results from: {results_dir_path}")
    
    loaded_data = load_results(results_dir_path, args.runs)
    
    if loaded_data:
        plot_combined_comparison(loaded_data, output_image_path, project_root)
    else:
        print("No data found to plot. Please check --results-dir and --runs arguments.")
