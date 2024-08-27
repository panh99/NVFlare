# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
from logging import ERROR
from pathlib import Path
from typing import List

import pathspec
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.superexec.executor import Executor as FlowerSuperExecExecutor
from flwr.superexec.executor import RunTracker
from typing_extensions import Optional, override

from nvflare import FedJob
from nvflare.app_opt.flower.controller import FlowerController
from nvflare.app_opt.flower.executor import FlowerExecutor
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session

SESSION_ARGS = {"username", "startup_kit_location", "debug", "timeout"}


def _get_job_name(publisher: str, app_name: str, app_version: str) -> str:
    """Generate the job name."""
    # Replace invalid characters
    app_name = app_name.replace(" ", "_")

    # Return the job name
    return f"{app_name}@{publisher}[{app_version}]"


def _load_gitignore(directory: Path) -> pathspec.PathSpec:
    """Load and parse .gitignore file, returning a pathspec."""
    gitignore_path = directory / ".gitignore"
    patterns = ["__pycache__/"]  # Default pattern
    if gitignore_path.exists():
        with open(gitignore_path, encoding="UTF-8") as file:
            patterns.extend(file.readlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _locate_all_files(directory: Path) -> List[Path]:
    """Locate all allowed files in the directory."""
    # Allowed extensions
    allowed_exts = ["py", "toml", "md"]

    # Load gitignore
    gitignore = _load_gitignore(directory)

    # Walk through the directory
    files = []
    for entry in (_ for ext in allowed_exts for _ in directory.rglob(f"*.{ext}")):
        if not gitignore.match_file(entry.relative_to(directory)):
            files.append(entry)
    return files


def _export_and_submit_job(session: Session, job: FedJob, export_to: str) -> None:
    """Export and submit a job via the secure session."""
    # Export job
    job_path = Path(export_to) / job.name
    job.export_job(job_path.parent)

    # Submit job
    session.submit_job(str(job_path))


class FlowerSuperExecPlugin(FlowerSuperExecExecutor):
    """POC engine executor for Flower SuperExec."""

    def __init__(self) -> None:
        self._sess = None
        self.job_dir = None
        self.flwr_dir = None

    @override
    def set_config(
        self,
        config: UserConfig,
    ) -> None:
        """Set executor config arguments.

        Parameters
        ----------
        config : UserConfig
            A dictionary for configuration values.
            Supported configuration key/value pairs:
            - "job-dir": str
                The directory to which jobs are exported.
            - "flwr-dir": str
                The path to the Flower directory.
        """
        print(f"Setting config: {config}")
        if not config:
            return
        if job_dir := config.get("job-dir"):
            if not isinstance(job_dir, str):
                raise ValueError("The `job-dir` value should be of type `str`.")
            self.job_dir = job_dir
        if flwr_dir := config.get("flwr-dir"):
            if not isinstance(flwr_dir, str):
                raise ValueError("The `flwr-dir` value should be of type `str`.")
            self.flwr_dir = str(flwr_dir)

    @override
    def start_run(
        self,
        fab_file: bytes,
        override_config: UserConfig,
        federation_config: UserConfig,
    ) -> Optional[RunTracker]:
        """Start run using Flare Engine."""
        try:
            # Load FAB file and extract metadata
            fab_id, fab_version = get_fab_metadata(fab_file)

            # Install FAB
            fab_path = install_from_fab(fab_file, None, True)

            # Generate the job name
            publisher, app_name = fab_id.split("/")
            job_name = _get_job_name(publisher, app_name, fab_version)

            # Create FedJob
            job = FedJob(name=job_name)

            # Locate all allowed files in the FAB directory
            files = _locate_all_files(Path(fab_path))

            # TODO: send override_config to Server Job and then to the SuperLink
            # Define the controller workflow and send to server
            controller = FlowerController()
            job.to_server(controller)

            # Add flwr server code
            for file in files:
                job.to_server(str(file.absolute()))

            # Add clients
            executor = FlowerExecutor()
            job.to_clients(executor)
            for file in files:
                job.to_clients(str(file.absolute()))

            # Export & submit the job
            if self.job_dir is not None:
                _export_and_submit_job(self.sess(federation_config), job, self.job_dir)
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    _export_and_submit_job(self.sess(federation_config), job, tmpdir)

            # TODO: Return RunTracker
            return RunTracker(run_id=0, proc=None)  # Replace with actual run_id and proc
        except Exception:
            import traceback

            log(ERROR, "Could not start run: %s", traceback.format_exc())
            return None

    def sess(self, configs: UserConfig) -> Session:
        """Obtain a secure session."""
        # Check the validity of configs
        configs = {k: v for k, v in configs.items() if k in SESSION_ARGS}

        # Check if a session is already created
        if self._sess is not None:
            # If the args are the same, return the existing session
            if self.sess_cfgs == configs:
                return self._sess
            # Close the existing session
            self._sess.close()

        # Open a new session
        self._sess = new_secure_session(**configs)
        self.sess_cfgs = configs

        return self._sess


executor = FlowerSuperExecPlugin()
