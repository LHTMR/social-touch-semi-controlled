import argparse
import os
import logging
from pathlib import Path
from datetime import datetime
import shutil
import time
import traceback
from multiprocessing import Queue, freeze_support
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import open3d as o3d
# sklearn is assumed to be present in the Anaconda environment
from sklearn.decomposition import PCA

from prefect import flow

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mocking utils for template completeness - Replace with actual imports
from utils import path_tools
from utils import (
    DagConfigHandler,
    PipelineMonitor,
    TaskExecutor
)
from utils.pipeline.session_config_resolver import resolve_session_configs
from utils.should_process_task import should_process_task, clean_task_outputs
from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)


from _5_postprocessing import (
    fetch_forearm_of_reference,
    apply_icp_registration,
    calibrate_pca_xyz,
    project_contacts_onto_forearm,
    center_on_receptive_field,
    export_manual_label_ref_point,
    deduplicate_forearm_ply,
    deduplicate_contact_points_csv,
    monitor_deduplicate_xy_interactive,
)
from _5_postprocessing.set_xyz_reference_from_gestures import CalibrationConfig
from _4_merging.aggregate_blocks_session import aggregate_session_blocks

# --- Post-Processing Sub-Flows ---

@flow(name="fetch_forearm_of_reference")
def fetch_forearm_of_reference_flow(
    session_configs: List[KinectConfig],
    output_dir: Path,
    force_processing: bool = False,
) -> Path:
    """Fetch and stage the session forearm PLY into forearm_source/."""
    print(f"[{output_dir.name}] Fetching forearm-of-reference PLY...")
    return fetch_forearm_of_reference(
        session_configs, output_dir,
        force_processing=force_processing,
    )


@flow(name="apply_icp_registration")
def apply_icp_registration_flow(
    input_files: List[Path],
    session_configs: List[KinectConfig],
    output_dir: Path,
    force_processing: bool = False,
) -> List[Path]:
    """Apply ICP registration transforms to merged CSVs."""
    print(f"[{output_dir.name}] Applying ICP registration to {len(input_files)} files...")
    return apply_icp_registration(
        input_files, session_configs, output_dir,
        force_processing=force_processing,
    )

@flow(name="calibrate_pca_xyz")
def calibrate_pca_xyz_flow(
    input_files: List[Path],
    output_dir: Path,
    forearm_ply_path: Path,
    forearm_output_dir: Path,
    force_processing: bool = False,
) -> Tuple[List[Path], Path, Path]:
    """Apply PCA calibration to block CSVs and the forearm PLY."""
    print(f"[{output_dir.name}] Calibrating PCA XYZ reference on {len(input_files)} files...")
    return calibrate_pca_xyz(
        input_files, output_dir, forearm_ply_path, forearm_output_dir,
        monitor=False,
        monitor_segment=False,
        force_processing=force_processing,
    )


@flow(name="deduplicate_xy")
def deduplicate_xy_flow(
    input_files: List[Path],
    forearm_ply_path: Path,
    output_dir: Path,
    forearm_output_dir: Path,
    force_processing: bool = False,
    monitor: bool = True,
    epsilon: float = 5.0,
) -> Tuple[List[Path], Optional[Path]]:
    """Deduplicate the unified forearm PLY and contact points in registered CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    forearm_output_dir.mkdir(parents=True, exist_ok=True)

    expected_output_csvs = [output_dir / f.name for f in input_files]
    expected_forearm_out = forearm_output_dir / forearm_ply_path.name
    all_outputs = expected_output_csvs + [expected_forearm_out]

    if not should_process_task(
        input_paths=list(input_files) + [forearm_ply_path],
        output_paths=all_outputs,
        force=force_processing,
    ):
        logging.info("Deduplication up-to-date. Skipping.")
        return expected_output_csvs, expected_forearm_out

    clean_task_outputs(all_outputs)

    if monitor:
        try:
            pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
            vertices = np.asarray(pcd.points, dtype=np.float64)
            if len(vertices) > 0:
                epsilon = monitor_deduplicate_xy_interactive(vertices, initial_epsilon=epsilon)
                logging.info("User selected epsilon = %.4f from interactive monitor.", epsilon)
            else:
                logging.warning("Forearm PLY has 0 points, skipping monitor.")
        except KeyboardInterrupt:
            logging.info("Monitor aborted by user. Using default epsilon=%.4f.", epsilon)

    # Deduplicate the single forearm PLY
    forearm_out = forearm_output_dir / forearm_ply_path.name
    stats = deduplicate_forearm_ply(forearm_ply_path, forearm_out, epsilon=epsilon)
    logging.info(
        "Forearm dedup: %d → %d (removed %d)",
        stats["n_original"],
        stats["n_deduped"],
        stats["n_removed"],
    )

    # Deduplicate contact points in each registered CSV
    deduped_csv_paths: List[Path] = []
    for input_csv in input_files:
        csv_out = output_dir / input_csv.name
        stats = deduplicate_contact_points_csv(input_csv, csv_out, epsilon=epsilon)
        logging.info(
            "CSV dedup (%s): %d rows, %d → %d contact points",
            input_csv.stem,
            stats["n_rows_processed"],
            stats["total_points_before"],
            stats["total_points_after"],
        )
        deduped_csv_paths.append(csv_out)

    return deduped_csv_paths, forearm_out


@flow(name="project_contacts_onto_registered_forearm")
def project_contacts_onto_registered_forearm_flow(
    input_files: List[Path],
    forearm_ply_path: Path,
    output_dir: Path,
    projection_stats_path: Path,
    force_processing: bool = False,
) -> List[Path]:
    """Project contact points onto the deduplicated forearm surface.

    Args:
        input_files: Per-block deduplicated CSVs from ``blocks_registered_deduped/``.
        forearm_ply_path: Deduplicated forearm PLY to project onto.
        output_dir: Destination directory (``blocks_registered_projected/``).
        projection_stats_path: Path for the combined projection-stats CSV.
        force_processing: Re-run even when outputs are already up-to-date.

    Returns:
        List of output CSV paths in *output_dir*.
    """
    print(f"[{output_dir.name}] Projecting {len(input_files)} blocks onto forearm surface...")

    output_files = project_contacts_onto_forearm(
        input_files=input_files,
        forearm_ply_path=forearm_ply_path,
        output_dir=output_dir,
        projection_stats_path=projection_stats_path,
        force_processing=force_processing,
    )
    if not output_files:
        raise RuntimeError(
            f"project_contacts_onto_forearm returned no outputs — "
            f"expected {len(input_files)} output CSVs."
        )
    return output_files


@flow(name="center_on_receptive_field")
def center_on_receptive_field_flow(
    input_files: List[Path],
    forearm_ply_path: Optional[Path],
    output_dir: Path,
    forearm_output_dir: Path,
    rf_origin_path: Path,
    force_processing: bool = False,
) -> List[Path]:
    """Center block CSVs and forearm PLY on the receptive field origin."""
    print(f"[{output_dir.name}] Centering spatial data on receptive field origin...")
    return center_on_receptive_field(
        input_files, forearm_ply_path, output_dir, forearm_output_dir, rf_origin_path,
        force_processing=force_processing,
    )


@flow(name="aggregate_session_blocks")
def aggregate_session_blocks_flow(
    input_files: List[Path],
    output_path: Path,
    forearm_ply_path: Optional[Path] = None,
    force_processing: bool = False,
) -> Path:
    """Aggregate all fully-processed block CSVs into one session-level CSV."""
    print(f"[{output_path.name}] Aggregating {len(input_files)} blocks...")
    return aggregate_session_blocks(
        input_paths=input_files,
        output_path=output_path,
        forearm_ply_path=forearm_ply_path,
        force_processing=force_processing,
    )


@flow(name="export_manual_label_ref_point")
def export_manual_label_ref_point_flow(
    session_configs: List[KinectConfig],
    pca_calibration_path: Path,
    rf_origin_path: Path,
    forearm_ply_path: Path,
    output_path: Path,
    force_processing: bool = False,
) -> Path:
    """Export the manually labelled forearm reference point in the RF-centred frame."""
    print(f"[{output_path.parent.name}] Exporting the manual forearm reference point...")
    return export_manual_label_ref_point(
        session_configs, pca_calibration_path, rf_origin_path, forearm_ply_path, output_path,
        force_processing=force_processing,
    )


# --- Worker Flow ---

# @flow(name="Run Single Session Postprocessing")
def run_single_session_postprocessing(
    session_id: str,
    session_configs: List[KinectConfig],
    dag_handler: DagConfigHandler,
    *,
    monitor_queue: Queue = None,
    report_file_path: Path = None
):
    print(f"🚀 Starting postprocessing for session: {session_id}")
    # Assuming all configs in a session share the same merged output dir root
    session_output_dir = session_configs[0].session_merged_output_dir

    # Resolve input files from Configs
    # We look for the specific file expected from the video processing stage
    session_input_files = []
    for config in session_configs:
        input_dir = config.session_merged_output_dir / "blocks_filtered"
        input_path = input_dir / f"{config.session_id}_semicontrolled_{config.block_id}_merged_data.csv"
        if not input_path.exists():
            raise FileNotFoundError(
                f"Filtered merged block CSV not found: {input_path}. "
                f"Run the merging pipeline's filter_by_neural_quality task to "
                f"produce blocks_filtered/ before running postprocessing."
            )
        session_input_files.append(input_path)

    if monitor_queue is not None:
        monitor = PipelineMonitor(
            report_path=report_file_path, stages=list(dag_handler.tasks.keys()), data_queue=monitor_queue
        )
    else:
        monitor = None

    # Initialize context with the raw list of files AND the session configs
    context = {
        "source_files": session_input_files,
        "session_configs": session_configs
    }

    # Pipeline stages in execution order
    pipeline_stages = [
        # Step 0: Fetch forearm-of-reference — copies the canonical forearm PLY into forearm_source/
        {
            "name": "fetch_forearm_of_reference",
            "func": fetch_forearm_of_reference_flow,
            "params": lambda: {
                "session_configs": context.get("session_configs"),
                "output_dir": session_output_dir / "forearm_source",
            },
            "outputs": ["source_forearm"]
        },
        # Step 1: ICP Registration — aligns all blocks into a common coordinate frame
        {
            "name": "apply_icp_registration",
            "func": apply_icp_registration_flow,
            "params": lambda: {
                "input_files": context.get("source_files"),
                "session_configs": context.get("session_configs"),
                "output_dir": session_output_dir / "blocks_registered",
            },
            "outputs": ["registered_files"]
        },
        # Step 2: Deduplicate (x,y) in the unified forearm PLY and contact CSVs
        {
            "name": "deduplicate_xy",
            "func": deduplicate_xy_flow,
            "params": lambda: {
                "input_files": context.get("registered_files"),
                "forearm_ply_path": context.get("source_forearm"),
                "output_dir": session_output_dir / "blocks_deduped",
                "forearm_output_dir": session_output_dir / "forearm_deduped",
            },
            "outputs": ["deduped_files", "deduped_forearm"]
        },
        # Step 3: Project contact points onto the deduplicated forearm surface
        {
            "name": "project_contacts_onto_forearm",
            "func": project_contacts_onto_registered_forearm_flow,
            "params": lambda: {
                "input_files": context.get("deduped_files"),
                "forearm_ply_path": context.get("deduped_forearm"),
                "output_dir": session_output_dir / "blocks_projected",
                "projection_stats_path": session_output_dir / "blocks_projected" / "projection_stats.csv",
            },
            "outputs": ["projected_files"]
        },
        # Step 4: PCA XYZ Calibration — applies PCA transform to CSVs and forearm PLY
        {
            "name": "calibrate_pca_xyz",
            "func": calibrate_pca_xyz_flow,
            "params": lambda: {
                "input_files": context.get("projected_files"),
                "output_dir": session_output_dir / "blocks_pca_calibrated",
                "forearm_ply_path": context.get("deduped_forearm"),
                "forearm_output_dir": session_output_dir / "forearm_pca_calibrated",
            },
            "outputs": ["pca_files", "pca_output_dir", "pca_forearm"]
        },
        # Step 5: Center spatial data on the receptive field origin
        {
            "name": "center_on_receptive_field",
            "func": center_on_receptive_field_flow,
            "params": lambda: {
                "input_files": context.get("pca_files"),
                "forearm_ply_path": context.get("pca_forearm"),
                "output_dir": session_output_dir / "blocks_rf_centered",
                "forearm_output_dir": session_output_dir / "forearm_rf_centered",
                "rf_origin_path": session_output_dir / "rf_center_origin.json",
            },
            "outputs": ["rf_files"]
        },
        # Step 6: Aggregate fully-processed blocks into one session-level CSV
        {
            "name": "aggregate_session",
            "func": aggregate_session_blocks_flow,
            "params": lambda: {
                "input_files": context.get("rf_files"),
                "output_path": session_output_dir / f"{session_id}_semicontrolled_aggregated_session.csv",
                "forearm_ply_path": session_output_dir / "forearm_rf_centered" / context.get("pca_forearm").name
                    if context.get("pca_forearm") is not None
                    else None,
            },
            "outputs": ["aggregated_file"]
        },
        # Step 7: Export the manually labelled forearm reference point (black dot)
        {
            "name": "export_manual_label_ref_point",
            "func": export_manual_label_ref_point_flow,
            "params": lambda: {
                "session_configs": context.get("session_configs"),
                "pca_calibration_path": context.get("pca_output_dir") / CalibrationConfig.output_json_name,
                "rf_origin_path": session_output_dir / "rf_center_origin.json",
                "forearm_ply_path": session_output_dir / "forearm_rf_centered" / context.get("pca_forearm").name,
                "output_path": session_output_dir / "manual_label_ref_point.json",
            },
            "outputs": ["manual_label_ref_point"]
        },
    ]

    for stage in pipeline_stages:
        task_name = stage["name"]
        # Use session_id as block_name equivalent here for the executor
        executor = TaskExecutor(task_name, session_id, dag_handler, monitor)

        with executor:
            if not executor.can_run: continue
            
            # Retrieve options from DAG handler (e.g. force_processing)
            options = dag_handler.get_task_options(task_name)
            
            # Resolve parameters lazily from context
            try:
                params = stage["params"]()
            except KeyError as e:
                 executor.error_msg = f"Missing dependency in context: {e}"
                 print(f"❌ {executor.error_msg}")
                 # Logic for failure inside executor context usually requires setting the error 
                 # and allowing the __exit__ to handle logging
                 continue

            # Inject options into params when supported by the flow
            if 'force_processing' in options:
                params['force_processing'] = options['force_processing']
            if 'monitor' in options:
                params['monitor'] = options['monitor']
            if 'epsilon' in options:
                params['epsilon'] = options['epsilon']
            
            # Validation: Check if list inputs are empty
            # Note: We must exclude 'configs' from this check if configs are not lists of files, 
            # though in this architecture they are a List[KinectConfig], so len() check is valid.
            input_lists = [v for v in params.values() if isinstance(v, list)]
            if any(len(l) == 0 for l in input_lists):
                 print(f"⚠️ Warning: Empty input list for task {task_name}. Skipping execution.")
                 continue

            result = stage["func"](**params)

            if "outputs" in stage:
                outputs = stage["outputs"]
                # Ensure result is iterable/tuple for unpacking
                if not isinstance(result, tuple): 
                    result = (result,)
                
                for i, key in enumerate(outputs):
                    if key and i < len(result): 
                        context[key] = result[i]

        if executor.error_msg:
            print(f"🛑 Failure in {task_name}. Aborting session.")
            return {"status": "failed", "error": executor.error_msg}

    print(f"✅ Postprocessing finished for: {session_id}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}

def run_batch_postprocessing(
    block_files: list[Path],
    project_data_root: Path,
    dag_config_path: Path,
    monitor_queue: Queue,
    report_file_path: Path,
    parallel: bool
):
    """
    Loads all block configurations, groups them by session_id, and triggers postprocessing per session.
    """
    # 1. Load Configs
    dag_handler_template = DagConfigHandler(dag_config_path)
    if not block_files:
        logging.warning("No config files found.")
        return

    # 2. Group by Session ID
    # Dictionary structure: { "session_id": [KinectConfig_Block1, KinectConfig_Block2, ...] }
    session_map: Dict[str, List[KinectConfig]] = defaultdict(list)
    
    logging.info(f"Scanning {len(block_files)} config files...")
    
    for block_file in block_files:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            # Initialize config object to resolve paths and IDs
            config = KinectConfig(config_data=config_data, database_path=project_data_root)
            session_map[config.session_id].append(config)
        except Exception as e:
            logging.error(f"Failed to load config {block_file}: {e}")

    logging.info(f"Found {len(session_map)} unique sessions to process.")

    # 3. Execute Pipeline per Session
    dag_handler_template = DagConfigHandler(dag_config_path)
    
    mode = "PARALLEL" if parallel else "SEQUENTIAL"
    logging.info(f"🚀 Starting postprocessing batch in {mode} mode.")
    
    for session_id, session_configs in session_map.items():
        dag_handler_instance = dag_handler_template.copy()
        
        if parallel:
            # Prefect Future submission logic would go here
            pass
        else:
            run_single_session_postprocessing(
                session_id=session_id,
                session_configs=session_configs,
                dag_handler=dag_handler_instance,
                monitor_queue=monitor_queue,
                report_file_path=report_file_path
            )

    logging.info("✅ All postprocessing tasks finished.")

# --- Main ---

def setup_environment():
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")

    print("🛠️  Setting up environment...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_file_path = reports_dir / f"{timestamp}_postprocess_workflow_kinect_auto_status.xlsx"
    if report_file_path.exists():
        report_file_path.unlink()
        print("🧹 File with the same name found, removing it.")
    return project_data_root, configs_dir, report_file_path

def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-config", type=Path, required=True)
    args = parser.parse_args()
    dag_config_path = args.dag_config
    project_data_root, configs_dir, report_file_path = setup_environment()

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        is_parallel = main_dag_handler.get_parameter('parallel_execution', False)
        entries = main_dag_handler.get_parameter('kinect_configs')
        block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{dag_config_path}' not found.")
        exit(1)

    print("📊 Initializing pipeline monitor...")
    pipeline_stages = list(main_dag_handler.tasks.keys())
    main_monitor = PipelineMonitor(report_path=str(report_file_path), stages=pipeline_stages, live_plotting=True)
    main_monitor.show_dashboard()

    try:
        run_batch_postprocessing(
            block_files=block_files,
            project_data_root=project_data_root,
            dag_config_path=dag_config_path,
            monitor_queue=main_monitor.queue,
            report_file_path=report_file_path,
            parallel=is_parallel,
        )
        print("\n🏁 All pipeline tasks have completed.")
        print("✨ Dashboard will close automatically in 10 seconds...")
        time.sleep(10)
    finally:
        main_monitor.close_dashboard(block=True)
        print(f"👋 Processing finished. Final report saved to {report_file_path}")

if __name__ == "__main__":
    main()