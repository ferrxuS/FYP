import streamlit as st
import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# Import fyp modules safely
from fyp.models.segmenter import ContainerDefectSegmenter
from fyp.config import (
    SAVE_DIR, RESULTS_DIR, DEVICE, BACKBONE, NUM_CLASSES, 
    DECODER_HIDDEN_DIM, VALID_DIR, IMAGE_SIZE
)
from fyp.data.dataset import ContainerDefectDataset

# Page setup with wide, premium layout
st.set_page_config(
    page_title="Continual Learning Container Defect Segmentation Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling for rich aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 15px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: bold;
        color: #1F2937;
    }
    .metric-label {
        font-size: 14px;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .card-si { border-left-color: #EF4444; }
    .card-mer { border-left-color: #10B981; }
    .card-der { border-left-color: #8B5CF6; }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">🏗️ Continual Learning Container Defect Segmenter</div>', unsafe_allow_html=True)
subtitle_placeholder = st.empty()

# Helper function to scan available runs
def get_available_runs():
    runs = []
    if SAVE_DIR.exists():
        for path in SAVE_DIR.iterdir():
            if path.is_dir() and path.name.startswith("run_"):
                runs.append(path.name)
    return sorted(runs, key=lambda x: int(x.split("_")[1]), reverse=True)

# Helper function to check run type (Frozen vs. Unfrozen)
@st.cache_data
def get_run_type(run_name):
    run_dir = SAVE_DIR / run_name
    if not run_dir.exists():
        return "Unknown"
    
    pth_files = list(run_dir.glob("*.pth"))
    if not pth_files:
        return "Frozen"
        
    try:
        # Load weights on CPU to check keys without consuming GPU memory
        state_dict = torch.load(pth_files[0], map_location="cpu")
        has_backbone = any("backbone" in k for k in state_dict.keys())
        return "Unfrozen" if has_backbone else "Frozen"
    except Exception:
        # Fallback to file size check
        size_mb = pth_files[0].stat().st_size / (1024 * 1024)
        return "Unfrozen" if size_mb > 50.0 else "Frozen"

# Helper function to load results
def load_run_results(run_name):
    run_results_dir = RESULTS_DIR / run_name
    results = {}
    if run_results_dir.exists():
        for file in run_results_dir.iterdir():
            if file.name.startswith("cl_results_") and file.suffix == ".json":
                strategy = file.name.replace("cl_results_", "").replace(".json", "")
                try:
                    with open(file, "r") as f:
                        results[strategy] = json.load(f)
                except Exception as e:
                    pass
    return results

# Helper function to load selected model weights
@st.cache_resource(max_entries=1)
def load_pytorch_model(run_name, checkpoint_name):
    checkpoint_path = SAVE_DIR / run_name / checkpoint_name
    model = ContainerDefectSegmenter(
        backbone_name=BACKBONE,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        decoder_hidden_dim=DECODER_HIDDEN_DIM
    )
    try:
        # Load weights on cpu or device (strict=False supports pruned checkpoints)
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# Sidebar controls
st.sidebar.markdown("### ⚙️ Benchmark Controls")
runs = get_available_runs()
if not runs:
    st.sidebar.error("No training runs found in `models/` folder. Please execute training first.")
    st.stop()

# Selection Mode
selection_mode = st.sidebar.radio(
    "Run Selection Mode",
    ["Categorized (Frozen vs. Unfrozen)", "Direct Run Folder"],
    help="Select how you want to filter and search for your experiment runs."
)

if selection_mode == "Categorized (Frozen vs. Unfrozen)":
    backbone_status = st.sidebar.selectbox("Backbone Status", ["Frozen", "Unfrozen"])
    
    # 1. Find all runs matching this backbone status
    matching_runs = []
    for r in runs:
        if get_run_type(r) == backbone_status:
            matching_runs.append(r)
            
    if not matching_runs:
        st.sidebar.warning(f"No runs found categorized as {backbone_status}")
        st.stop()
        
    # 2. Find all unique strategies across these matching runs
    strategy_to_runs = {}
    for r in matching_runs:
        run_res = load_run_results(r)
        for strat in run_res.keys():
            strat_lower = strat.lower()
            if strat_lower not in strategy_to_runs:
                strategy_to_runs[strat_lower] = []
            strategy_to_runs[strat_lower].append(r)
            
    if not strategy_to_runs:
        st.sidebar.warning(f"No strategy results found for {backbone_status} runs.")
        st.stop()
        
    # Mapping display names to keys
    strategy_display_names = {
        'naive': 'Naive Fine-tuning',
        'er': 'Experience Replay (ER)',
        'der++': 'Dark Experience Replay (DER++)',
        'mer': 'Meta Experience Replay (MER)',
        'si': 'Synaptic Intelligence (SI)',
        'lwf': 'Learning Without Forgetting (LwF)',
        'slca': 'Slow Learner Classifier Alignment (SLCA)',
        'er_ace': 'Experience Replay ACE (ER-ACE)'
    }
    
    available_strat_keys = sorted(list(strategy_to_runs.keys()))
    strat_options = []
    strat_key_map = {}
    for k in available_strat_keys:
        disp = strategy_display_names.get(k, k.upper())
        strat_options.append(disp)
        strat_key_map[disp] = k
        
    selected_strat_disp = st.sidebar.selectbox("Select Model / Strategy", strat_options)
    selected_strat_key = strat_key_map[selected_strat_disp]
    
    # Select the latest run containing this strategy (runs are sorted descending)
    selected_run = strategy_to_runs[selected_strat_key][0]
    run_type = backbone_status
    
    results = load_run_results(selected_run)
    
    selected_strategy = None
    for k in results.keys():
        if k.lower() == selected_strat_key:
            selected_strategy = k
            break
            
    # Filter to only the selected strategy in Categorized mode
    if selected_strategy:
        available_strategies = [selected_strategy]
        results = {selected_strategy: results[selected_strategy]}
    else:
        available_strategies = list(results.keys())
else:
    selected_run = st.sidebar.selectbox("Select Experiment Run", runs)
    run_type = get_run_type(selected_run)
    
    results = load_run_results(selected_run)
    available_strategies = list(results.keys())
    if not available_strategies:
        st.sidebar.warning(f"No completed CL JSON results found for {selected_run}")
        selected_strategy = None
    else:
        selected_strategy = st.sidebar.selectbox("Active Strategy for Analytics", available_strategies)

# Render subtitle dynamically based on selected run type
subtitle_placeholder.markdown(
    f'<div class="subtitle">FYP Evaluation and Interactive Inference System ({run_type} DINOv3 Backbone)</div>', 
    unsafe_allow_html=True
)

# Tab separation
tab_analytics, tab_inference = st.tabs(["CL Analytics & Comparisons", "Inference Playpen"])

# Category descriptions for defect/repair annotations
DEFECT_CATEGORIES = {
    0: "Background",
    1: "Paint Repair (PAINT-P)",
    2: "Patch Repair (PATCH-PT)",
    3: "Removal Repair (REMOVE-R)",
    4: "Sealant Repair (SEALANT-SE)",
    5: "Wash Repair (WASH-W)"
}

# --- TAB 1: ANALYTICS ---
with tab_analytics:
    if not selected_strategy:
        st.info("No comparative analytics available for this run directory yet.")
    else:
        st.markdown(f"### Comparative Overview for **{selected_run.upper()}**")
        
        # Strategy overview cards side-by-side
        cols = st.columns(len(available_strategies))
        for i, strat in enumerate(available_strategies):
            strat_data = results[strat]['metrics']
            avg_acc = strat_data.get('Average Accuracy', 0)
            forgetting = strat_data.get('Forgetting', 0)
            
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card card-{strat}">
                    <div class="metric-label">{strat.upper()}</div>
                    <div class="metric-val">{avg_acc:.4f}</div>
                    <div class="metric-label">Avg mIoU</div>
                    <div style="margin-top: 10px;" class="metric-label">Forgetting: {forgetting:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Double column layout for accuracy heatmaps and metrics matrices
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"#### Accuracy Heatmap Matrix - {selected_strategy.upper()}")
            matrix = np.array(results[selected_strategy]['metrics']['Accuracy Matrix'])
            
            fig, ax = plt.subplots(figsize=(7, 5))
            mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
            sns.heatmap(
                matrix,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'mIoU'},
                square=True,
                ax=ax
            )
            ax.set_title(f'Accuracy Matrix: {selected_strategy.upper()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Task Evaluated On')
            ax.set_ylabel('After Training Task')
            tick_labels = [f'Task {i+1}' for i in range(matrix.shape[0])]
            ax.set_xticklabels(tick_labels)
            ax.set_yticklabels(tick_labels, rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        with col_right:
            st.markdown("#### Task Performance Retention Curves")
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            num_tasks = matrix.shape[0]
            
            for task_id in range(num_tasks):
                performance = []
                x_points = []
                for train_task in range(task_id, num_tasks):
                    if matrix[train_task, task_id] > 0:
                        performance.append(matrix[train_task, task_id])
                        x_points.append(train_task + 1)
                
                ax2.plot(
                    x_points,
                    performance,
                    marker='o',
                    linewidth=2.5,
                    markersize=6,
                    label=f'Task {task_id + 1} ({DEFECT_CATEGORIES[task_id+1]})'
                )
            
            ax2.set_xlabel('After Training Task')
            ax2.set_ylabel('mIoU Score')
            ax2.set_title('Retrospective Performance Curve', fontsize=12, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(1, num_tasks + 1))
            ax2.set_ylim([0, 1])
            plt.tight_layout()
            st.pyplot(fig2)

# --- TAB 2: INFERENCE PLAYPEN ---
with tab_inference:
    st.markdown("### Real-Time Defect Segmentation Inference Playpen")
    st.write("Test your trained continual learning models dynamically on validation images.")

    # Model checkpoints selection
    run_path = SAVE_DIR / selected_run
    checkpoints = [p.name for p in run_path.iterdir() if p.suffix == ".pth"]
    
    if not checkpoints:
        st.info("No trained checkpoint `.pth` files found in this run directory.")
    else:
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            selected_ckpt = st.selectbox("Select Model Checkpoint (.pth)", sorted(checkpoints))
        with col_c2:
            conf_threshold = st.slider("Prediction Threshold Filter", 0.0, 1.0, 0.5, 0.05)

        # Load validation dataset dynamically
        if not VALID_DIR.exists():
            st.error(f"Validation directory not found at {VALID_DIR}")
        else:
            # Load dataset cleanly
            @st.cache_resource
            def get_cached_dataset():
                return ContainerDefectDataset(VALID_DIR)
            
            dataset = get_cached_dataset()
            
            if "image_index_val" not in st.session_state:
                st.session_state["image_index_val"] = 0

            current_idx = st.session_state["image_index_val"]

            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                image_idx = st.number_input(
                    f"Select Image Index (0 to {len(dataset)-1})",
                    min_value=0, max_value=len(dataset)-1,
                    value=current_idx
                )
                if image_idx != current_idx:
                    st.session_state["image_index_val"] = image_idx
                    st.rerun()
            with col_sel2:
                # Random selector button
                if st.button("Random Validation Image"):
                    st.session_state["image_index_val"] = int(np.random.randint(0, len(dataset)))
                    st.rerun()

            # Retrieve selected image and mask
            image, mask, filename = dataset[image_idx]

            # Trigger PyTorch model loading
            model, err = load_pytorch_model(selected_run, selected_ckpt)

            if err:
                st.error(f"Error loading model weights: {err}")
            else:
                # Run real-time inference
                image_input = image.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(image_input)
                    probabilities = F.softmax(logits, dim=1).cpu().squeeze(0)
                    predictions = torch.argmax(logits, dim=1).cpu().squeeze(0).numpy()

                # Process original image for visualization
                # Denormalize image back to range [0, 1]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_denorm = image * std + mean
                image_denorm = torch.clamp(image_denorm, 0, 1).permute(1, 2, 0).numpy()

                # Generate clean color map overlay for defect channels
                # Map colors: rust=red, structural=orange, paint=yellow, crack=purple, decal=blue
                colors = [
                    [0, 0, 0],       # Background (transparent)
                    [239, 68, 68],   # 1: Paint Repair (Red)
                    [245, 158, 11],  # 2: Patch Repair (Orange)
                    [252, 211, 77],  # 3: Removal Repair (Yellow)
                    [139, 92, 246],  # 4: Sealant Repair (Purple)
                    [59, 130, 246]   # 5: Wash Repair (Blue)
                ]

                # Create colored overlays
                mask_np = mask.squeeze(0).numpy()
                gt_overlay = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
                pred_overlay = np.zeros((*predictions.shape, 3), dtype=np.uint8)

                for c_id, color in enumerate(colors):
                    if c_id == 0: continue
                    gt_overlay[mask_np == c_id] = color
                    # For predictions, filter out by confidence threshold of softmax
                    conf_mask = (predictions == c_id) & (probabilities[c_id].numpy() >= conf_threshold)
                    pred_overlay[conf_mask] = color

                # Compute blending
                gt_blended = np.copy(image_denorm)
                pred_blended = np.copy(image_denorm)
                alpha = 0.55

                for c_id in range(1, len(colors)):
                    gt_class_mask = (mask_np == c_id)
                    pred_class_mask = (predictions == c_id) & (probabilities[c_id].numpy() >= conf_threshold)
                    
                    gt_blended[gt_class_mask] = (1 - alpha) * gt_blended[gt_class_mask] + alpha * (np.array(colors[c_id])/255.0)
                    pred_blended[pred_class_mask] = (1 - alpha) * pred_blended[pred_class_mask] + alpha * (np.array(colors[c_id])/255.0)

                # Show side-by-side visualization
                col_v1, col_v2, col_v3 = st.columns([1, 1, 1])

                with col_v1:
                    st.image(image_denorm, caption=f"Original Image: {filename}", use_container_width=True)
                with col_v2:
                    st.image(gt_blended, caption="Ground Truth Defect Overlay", use_container_width=True)
                with col_v3:
                    st.image(pred_blended, caption=f"Model Prediction ({selected_ckpt})", use_container_width=True)

                # Display Color Legend
                st.markdown("#### Color Legend Mapping")
                legend_cols = st.columns(5)
                for cat_id in range(1, 6):
                    with legend_cols[cat_id - 1]:
                        hex_color = '#{:02x}{:02x}{:02x}'.format(*colors[cat_id])
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div style="width: 20px; height: 20px; border-radius: 4px; background-color: {hex_color};"></div>
                            <span style="font-size: 13px; font-weight: 500;">{DEFECT_CATEGORIES[cat_id]}</span>
                        </div>
                        """, unsafe_allow_html=True)
