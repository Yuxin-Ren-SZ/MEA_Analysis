#!/usr/bin/env python3
"""
Tkinter GUI for run_pipeline_driver.py

A graphical interface for configuring and launching the MEA pipeline driver
without using the command line.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import threading
import sys
import os
from datetime import datetime
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MEA Pipeline Driver")
        self.root.geometry("800x900")
        self.root.minsize(700, 800)

        # Process reference for pipeline execution
        self.process = None
        self.is_running = False

        # Create main frame with scrollbar
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Build GUI sections
        self._build_path_section()
        self._build_file_inputs_section()
        self._build_options_section()
        self._build_checkbox_section()
        self._build_preview_section()
        self._build_action_section()
        self._build_log_section()

        # Initial preview update
        self.update_preview()

    def _build_path_section(self):
        """Build the required path inputs section."""
        frame = ttk.LabelFrame(self.main_frame, text="Required Paths", padding="5")
        frame.pack(fill=tk.X, pady=(0, 10))

        # Data Path
        ttk.Label(frame, text="Data Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.data_path_var = tk.StringVar()
        self.data_path_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Entry(frame, textvariable=self.data_path_var, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(frame, text="Browse File...", command=lambda: self._browse_file(self.data_path_var)).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(frame, text="Browse Dir...", command=lambda: self._browse_directory(self.data_path_var)).grid(row=0, column=3, padx=2, pady=2)

        # Output Directory
        ttk.Label(frame, text="Output Dir:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.output_dir_var = tk.StringVar()
        self.output_dir_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Entry(frame, textvariable=self.output_dir_var, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_directory(self.output_dir_var)).grid(row=1, column=2, padx=2, pady=2)

    def _build_file_inputs_section(self):
        """Build optional file inputs section."""
        frame = ttk.LabelFrame(self.main_frame, text="Optional File Inputs", padding="5")
        frame.pack(fill=tk.X, pady=(0, 10))

        # Reference Excel
        ttk.Label(frame, text="Reference Excel:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.reference_var = tk.StringVar()
        self.reference_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Entry(frame, textvariable=self.reference_var, width=50).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.reference_var, [("Excel files", "*.xlsx *.xls")])).grid(row=0, column=2, padx=2, pady=2)

        # Params JSON
        ttk.Label(frame, text="Params JSON:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.params_var = tk.StringVar()
        self.params_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Entry(frame, textvariable=self.params_var, width=50).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.params_var, [("JSON files", "*.json")])).grid(row=1, column=2, padx=2, pady=2)

        # Checkpoint Directory
        ttk.Label(frame, text="Checkpoint Dir:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.checkpoint_dir_var = tk.StringVar()
        self.checkpoint_dir_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Entry(frame, textvariable=self.checkpoint_dir_var, width=50).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_directory(self.checkpoint_dir_var)).grid(row=2, column=2, padx=2, pady=2)

    def _build_options_section(self):
        """Build dropdown and text options section."""
        frame = ttk.LabelFrame(self.main_frame, text="Processing Options", padding="5")
        frame.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Sorter and Docker
        ttk.Label(frame, text="Sorter:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.sorter_var = tk.StringVar(value="kilosort4")
        self.sorter_var.trace_add("write", lambda *args: self.update_preview())
        sorter_combo = ttk.Combobox(frame, textvariable=self.sorter_var, width=15, state="readonly")
        sorter_combo['values'] = ("kilosort2", "kilosort2_5", "kilosort3", "kilosort4")
        sorter_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(frame, text="Docker Image:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.docker_var = tk.StringVar()
        self.docker_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Entry(frame, textvariable=self.docker_var, width=30).grid(row=0, column=3, padx=5, pady=2)

        # Row 2: Assay Types
        ttk.Label(frame, text="Assay Types:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.assay_types_var = tk.StringVar(value="network today, network today/best")
        self.assay_types_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Entry(frame, textvariable=self.assay_types_var, width=50).grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=2)

    def _build_checkbox_section(self):
        """Build checkbox options section."""
        outer_frame = ttk.Frame(self.main_frame)
        outer_frame.pack(fill=tk.X, pady=(0, 10))

        # Execution Options
        exec_frame = ttk.LabelFrame(outer_frame, text="Execution Options", padding="5")
        exec_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 5))

        self.dry_run_var = tk.BooleanVar()
        self.dry_run_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(exec_frame, text="Dry Run", variable=self.dry_run_var).pack(anchor=tk.W)

        self.debug_var = tk.BooleanVar()
        self.debug_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(exec_frame, text="Debug Mode", variable=self.debug_var).pack(anchor=tk.W)

        self.force_restart_var = tk.BooleanVar()
        self.force_restart_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(exec_frame, text="Force Restart", variable=self.force_restart_var).pack(anchor=tk.W)

        self.clean_up_var = tk.BooleanVar()
        self.clean_up_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(exec_frame, text="Clean Up", variable=self.clean_up_var).pack(anchor=tk.W)

        # Processing Options
        proc_frame = ttk.LabelFrame(outer_frame, text="Processing Options", padding="5")
        proc_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5)

        self.skip_spikesorting_var = tk.BooleanVar()
        self.skip_spikesorting_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(proc_frame, text="Skip Spike Sorting", variable=self.skip_spikesorting_var).pack(anchor=tk.W)

        self.no_curation_var = tk.BooleanVar()
        self.no_curation_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(proc_frame, text="No Curation", variable=self.no_curation_var).pack(anchor=tk.W)

        self.export_to_phy_var = tk.BooleanVar()
        self.export_to_phy_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(proc_frame, text="Export to Phy", variable=self.export_to_phy_var).pack(anchor=tk.W)

        self.reanalyze_bursts_var = tk.BooleanVar()
        self.reanalyze_bursts_var.trace_add("write", lambda *args: self.update_preview())
        ttk.Checkbutton(proc_frame, text="Reanalyze Bursts", variable=self.reanalyze_bursts_var).pack(anchor=tk.W)

        # Analysis Selection frame
        analysis_frame = ttk.LabelFrame(outer_frame, text="Analysis Selection", padding="5")
        analysis_frame.grid(row=0, column=2, columnspan=2, sticky=tk.NSEW, padx=5)

        # Flag to prevent recursive updates
        self._updating_analysis = False

        # Preset dropdown
        preset_row = ttk.Frame(analysis_frame)
        preset_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(preset_row, text="Preset:").pack(side=tk.LEFT, padx=(0, 5))
        self.analysis_preset_var = tk.StringVar(value="default")
        self.analysis_preset_var.trace_add("write", self._on_preset_change)
        preset_combo = ttk.Combobox(preset_row, textvariable=self.analysis_preset_var, width=12, state="readonly")
        preset_combo['values'] = ("default", "all", "minimal", "none", "custom")
        preset_combo.pack(side=tk.LEFT)

        # Individual analysis checkboxes (opt-in style)
        self.probe_var = tk.BooleanVar(value=True)
        self.probe_var.trace_add("write", self._on_analysis_checkbox_change)
        ttk.Checkbutton(analysis_frame, text="Probe Maps", variable=self.probe_var).pack(anchor=tk.W)

        self.waveforms_var = tk.BooleanVar(value=True)
        self.waveforms_var.trace_add("write", self._on_analysis_checkbox_change)
        ttk.Checkbutton(analysis_frame, text="Waveforms", variable=self.waveforms_var).pack(anchor=tk.W)

        self.raster_var = tk.BooleanVar(value=True)
        self.raster_var.trace_add("write", self._on_analysis_checkbox_change)
        ttk.Checkbutton(analysis_frame, text="Raster Plots", variable=self.raster_var).pack(anchor=tk.W)

        self.burst_var = tk.BooleanVar(value=True)
        self.burst_var.trace_add("write", self._on_analysis_checkbox_change)
        ttk.Checkbutton(analysis_frame, text="Burst Analysis", variable=self.burst_var).pack(anchor=tk.W)

        self.spatial_var = tk.BooleanVar(value=False)
        self.spatial_var.trace_add("write", self._on_analysis_checkbox_change)
        ttk.Checkbutton(analysis_frame, text="Spatial Maps", variable=self.spatial_var).pack(anchor=tk.W)

        # Preview label showing resolved analysis string
        self.analysis_preview_var = tk.StringVar(value="probe,waveforms,raster,burst")
        preview_label = ttk.Label(analysis_frame, textvariable=self.analysis_preview_var,
                                  foreground="gray", font=("TkDefaultFont", 9))
        preview_label.pack(anchor=tk.W, pady=(5, 0))

        # Configure column weights for even distribution
        outer_frame.columnconfigure(0, weight=1)
        outer_frame.columnconfigure(1, weight=1)
        outer_frame.columnconfigure(2, weight=2)

    def _build_preview_section(self):
        """Build command preview section."""
        frame = ttk.LabelFrame(self.main_frame, text="Command Preview", padding="5")
        frame.pack(fill=tk.X, pady=(0, 10))

        self.preview_text = scrolledtext.ScrolledText(frame, height=3, wrap=tk.WORD, state=tk.DISABLED)
        self.preview_text.pack(fill=tk.X, expand=True)

    def _build_action_section(self):
        """Build run/stop buttons section."""
        frame = ttk.Frame(self.main_frame)
        frame.pack(fill=tk.X, pady=(0, 10))

        self.run_button = ttk.Button(frame, text="Run Pipeline", command=self.run_pipeline)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(frame, text="Stop", command=self.stop_pipeline, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.clear_log_button = ttk.Button(frame, text="Clear Log", command=self.clear_log)
        self.clear_log_button.pack(side=tk.RIGHT, padx=5)

    def _build_log_section(self):
        """Build log output section."""
        frame = ttk.LabelFrame(self.main_frame, text="Log Output", padding="5")
        frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(frame, height=20, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _browse_file(self, var, filetypes=None):
        """Open file browser dialog."""
        if filetypes is None:
            filetypes = [("All files", "*.*"), ("HDF5 files", "*.h5"), ("NWB files", "*.nwb")]

        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            var.set(filepath)

    def _browse_directory(self, var):
        """Open directory browser dialog."""
        dirpath = filedialog.askdirectory()
        if dirpath:
            var.set(dirpath)

    def _on_preset_change(self, *args):
        """Update checkboxes when preset changes."""
        if self._updating_analysis:
            return

        self._updating_analysis = True
        preset = self.analysis_preset_var.get()

        preset_mappings = {
            "default": {"probe": True, "waveforms": True, "raster": True, "burst": True, "spatial": False},
            "all": {"probe": True, "waveforms": True, "raster": True, "burst": True, "spatial": True},
            "minimal": {"probe": False, "waveforms": False, "raster": False, "burst": True, "spatial": False},
            "none": {"probe": False, "waveforms": False, "raster": False, "burst": False, "spatial": False},
        }

        if preset in preset_mappings:
            mapping = preset_mappings[preset]
            self.probe_var.set(mapping["probe"])
            self.waveforms_var.set(mapping["waveforms"])
            self.raster_var.set(mapping["raster"])
            self.burst_var.set(mapping["burst"])
            self.spatial_var.set(mapping["spatial"])

        self._update_analysis_preview()
        self._updating_analysis = False
        self.update_preview()

    def _on_analysis_checkbox_change(self, *args):
        """Switch to custom preset when checkbox manually changed."""
        if self._updating_analysis:
            return

        self._updating_analysis = True

        # If raster is checked, auto-enable burst (dependency)
        if self.raster_var.get() and not self.burst_var.get():
            self.burst_var.set(True)

        # Switch to custom preset if current state doesn't match any preset
        current_state = {
            "probe": self.probe_var.get(),
            "waveforms": self.waveforms_var.get(),
            "raster": self.raster_var.get(),
            "burst": self.burst_var.get(),
            "spatial": self.spatial_var.get(),
        }

        preset_mappings = {
            "default": {"probe": True, "waveforms": True, "raster": True, "burst": True, "spatial": False},
            "all": {"probe": True, "waveforms": True, "raster": True, "burst": True, "spatial": True},
            "minimal": {"probe": False, "waveforms": False, "raster": False, "burst": True, "spatial": False},
            "none": {"probe": False, "waveforms": False, "raster": False, "burst": False, "spatial": False},
        }

        matched_preset = "custom"
        for preset_name, mapping in preset_mappings.items():
            if current_state == mapping:
                matched_preset = preset_name
                break

        if self.analysis_preset_var.get() != matched_preset:
            self.analysis_preset_var.set(matched_preset)

        self._update_analysis_preview()
        self._updating_analysis = False
        self.update_preview()

    def _update_analysis_preview(self):
        """Update the analysis preview label."""
        analysis_str = self._get_analysis_string()
        if analysis_str:
            self.analysis_preview_var.set(f"--analysis {analysis_str}")
        else:
            self.analysis_preview_var.set("(no analyses)")

    def _get_analysis_string(self):
        """Build analysis string from current checkbox state."""
        preset = self.analysis_preset_var.get()

        # For known presets, just return the preset name
        if preset in ("default", "all", "minimal", "none"):
            return preset

        # For custom, build from checkboxes
        analyses = []
        if self.probe_var.get():
            analyses.append("probe")
        if self.waveforms_var.get():
            analyses.append("waveforms")
        if self.raster_var.get():
            analyses.append("raster")
        if self.burst_var.get():
            analyses.append("burst")
        if self.spatial_var.get():
            analyses.append("spatial")

        return ",".join(analyses) if analyses else "none"

    def build_command(self):
        """Build the CLI command from current GUI state."""
        cmd_parts = ["python3", str(BASE_DIR / "run_pipeline_driver.py")]

        # Required: data path
        data_path = self.data_path_var.get().strip()
        if data_path:
            cmd_parts.append(f'"{data_path}"')
        else:
            cmd_parts.append("<DATA_PATH>")

        # Optional file/dir inputs
        output_dir = self.output_dir_var.get().strip()
        if output_dir:
            cmd_parts.append(f'--output-dir "{output_dir}"')

        reference = self.reference_var.get().strip()
        if reference:
            cmd_parts.append(f'--reference "{reference}"')

        params = self.params_var.get().strip()
        if params:
            cmd_parts.append(f'--params "{params}"')

        checkpoint_dir = self.checkpoint_dir_var.get().strip()
        if checkpoint_dir:
            cmd_parts.append(f'--checkpoint-dir "{checkpoint_dir}"')

        # Sorter
        sorter = self.sorter_var.get()
        if sorter and sorter != "kilosort4":
            cmd_parts.append(f"--sorter {sorter}")

        # Docker
        docker = self.docker_var.get().strip()
        if docker:
            cmd_parts.append(f"--docker {docker}")

        # Assay types (only if different from default)
        assay_types = self.assay_types_var.get().strip()
        if assay_types and assay_types != "network today, network today/best":
            # Split by comma and add each as a separate --type argument
            types_list = [t.strip() for t in assay_types.split(",")]
            for t in types_list:
                cmd_parts.append(f'--type "{t}"')

        # Boolean flags
        if self.dry_run_var.get():
            cmd_parts.append("--dry")
        if self.debug_var.get():
            cmd_parts.append("--debug")
        if self.force_restart_var.get():
            cmd_parts.append("--force-restart")
        if self.clean_up_var.get():
            cmd_parts.append("--clean-up")
        if self.skip_spikesorting_var.get():
            cmd_parts.append("--skip-spikesorting")
        if self.no_curation_var.get():
            cmd_parts.append("--no-curation")
        if self.export_to_phy_var.get():
            cmd_parts.append("--export-to-phy")
        if self.reanalyze_bursts_var.get():
            cmd_parts.append("--reanalyze-bursts")

        # Analysis selection
        analysis = self._get_analysis_string()
        if analysis and analysis != "default":
            cmd_parts.append(f"--analysis {analysis}")

        return " ".join(cmd_parts)

    def update_preview(self):
        """Update the command preview text."""
        cmd = self.build_command()
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, cmd)
        self.preview_text.config(state=tk.DISABLED)

    def log_message(self, message, newline=True):
        """Append a message to the log widget."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if newline:
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        else:
            self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)  # Auto-scroll
        self.log_text.config(state=tk.DISABLED)

    def clear_log(self):
        """Clear the log output."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def run_pipeline(self):
        """Execute the pipeline command."""
        data_path = self.data_path_var.get().strip()
        if not data_path:
            self.log_message("ERROR: Data path is required!")
            return

        if not os.path.exists(data_path):
            self.log_message(f"ERROR: Data path does not exist: {data_path}")
            return

        # Build command as list for subprocess
        cmd_list = [sys.executable, str(BASE_DIR / "run_pipeline_driver.py"), data_path]

        # Add optional arguments
        output_dir = self.output_dir_var.get().strip()
        if output_dir:
            cmd_list.extend(["--output-dir", output_dir])

        reference = self.reference_var.get().strip()
        if reference:
            cmd_list.extend(["--reference", reference])

        params = self.params_var.get().strip()
        if params:
            cmd_list.extend(["--params", params])

        checkpoint_dir = self.checkpoint_dir_var.get().strip()
        if checkpoint_dir:
            cmd_list.extend(["--checkpoint-dir", checkpoint_dir])

        sorter = self.sorter_var.get()
        if sorter:
            cmd_list.extend(["--sorter", sorter])

        docker = self.docker_var.get().strip()
        if docker:
            cmd_list.extend(["--docker", docker])

        assay_types = self.assay_types_var.get().strip()
        if assay_types:
            types_list = [t.strip() for t in assay_types.split(",")]
            for t in types_list:
                cmd_list.extend(["--type", t])

        # Boolean flags
        if self.dry_run_var.get():
            cmd_list.append("--dry")
        if self.debug_var.get():
            cmd_list.append("--debug")
        if self.force_restart_var.get():
            cmd_list.append("--force-restart")
        if self.clean_up_var.get():
            cmd_list.append("--clean-up")
        if self.skip_spikesorting_var.get():
            cmd_list.append("--skip-spikesorting")
        if self.no_curation_var.get():
            cmd_list.append("--no-curation")
        if self.export_to_phy_var.get():
            cmd_list.append("--export-to-phy")
        if self.reanalyze_bursts_var.get():
            cmd_list.append("--reanalyze-bursts")

        # Analysis selection
        analysis = self._get_analysis_string()
        if analysis:
            cmd_list.extend(["--analysis", analysis])

        # Update UI state
        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.log_message("Starting pipeline...")
        self.log_message(f"Command: {' '.join(cmd_list)}")

        # Run in background thread
        thread = threading.Thread(target=self._run_subprocess, args=(cmd_list,), daemon=True)
        thread.start()

    def _run_subprocess(self, cmd_list):
        """Run subprocess and stream output to log."""
        try:
            self.process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Read output line by line
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                # Schedule GUI update in main thread
                self.root.after(0, self._append_log_line, line)

            self.process.stdout.close()
            return_code = self.process.wait()

            if return_code == 0:
                self.root.after(0, self.log_message, "Pipeline completed successfully!")
            else:
                self.root.after(0, self.log_message, f"Pipeline exited with code {return_code}")

        except Exception as e:
            self.root.after(0, self.log_message, f"ERROR: {str(e)}")

        finally:
            self.root.after(0, self._pipeline_finished)

    def _append_log_line(self, line):
        """Append a line to the log (called from main thread)."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _pipeline_finished(self):
        """Reset UI state when pipeline finishes."""
        self.is_running = False
        self.process = None
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def stop_pipeline(self):
        """Stop the running pipeline."""
        if self.process and self.is_running:
            self.log_message("Stopping pipeline...")
            self.is_running = False
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.log_message("Pipeline stopped.")


def main():
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
