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

import os
import sys
from pathlib import Path
from typing import List

import pathspec
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.common.config import get_project_config
from flwr.superexec.executor import Executor as FlowerSuperExecExecutor
from flwr.superexec.executor import RunTracker
from typing_extensions import Dict, Optional, override

from nvflare import FedJob
from nvflare.app_opt.flower.controller import FlowerController
from nvflare.app_opt.flower.executor import FlowerExecutor
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session


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
    # Load gitignore
    gitignore = _load_gitignore(directory)
    # allowed_extensions = {".py", ".toml", ".md"}
    allowed_extensions = {".py"}

    # Walk through the directory
    files = []
    for entry in (file for ext in allowed_extensions for file in directory.rglob(f"*{ext}")):
        if not gitignore.match_file(entry.relative_to(directory)):
            files.append(entry)
    return files


class FlareEngine(FlowerSuperExecExecutor):
    """Flare engine executor for Flower SuperExec."""

    _sess = None

    @override
    def set_config(self, config: Dict[str, str]) -> None:
        """Set executor config arguments."""
        if not config:
            return
        if superlink_address := config.get("superlink"):
            # TODO: log warning
            # No need to set superlink address
            # This should be managed by NVFlare system
            pass
        if root_certificates := config.get("root-certificates"):
            # TODO: log warning
            # No need to set root certificates
            # This should be managed by NVFlare system
            pass
        if flwr_dir := config.get("flwr-dir"):
            self.flwr_dir = flwr_dir

    @override
    def start_run(self, fab_file: bytes, override_config: Dict[str, str]) -> Optional[RunTracker]:
        """Start run using Flare Engine."""
        try:
            # Load FAB file and extract metadata
            fab_version, fab_id = get_fab_metadata(fab_file)

            # Install FAB
            fab_path = install_from_fab(fab_file, None, True)

            # Retrieve the FAB info
            print(f"FAB path: {fab_path}")
            config = get_project_config(fab_path)
            print(f"FAB config: {config}")
            server_app_ref = config["tool"]["flwr"]["app"]["components"]["serverapp"]
            client_app_ref = config["tool"]["flwr"]["app"]["components"]["clientapp"]

            # TODO: Allow for any number of clients
            n_clients = 2

            # Generate the job name
            publisher, app_name = fab_id.split("/")
            job_name = _get_job_name(publisher, app_name, fab_version)

            # Create FedJob
            job = FedJob(name=job_name)

            # Locate all allowed files in the FAB directory
            files = _locate_all_files(Path(fab_path))

            # Define the controller workflow and send to server
            controller = FlowerController(server_app=server_app_ref)
            job.to(controller, "server")

            # Add flwr server code
            for file in files:
                print(f"Sending to job: {file}")
                job.to(str(file.absolute()), "server")

            # Add clients
            for i in range(1, n_clients + 1):
                executor = FlowerExecutor(client_app=client_app_ref)
                job.to(executor, f"site-{i}", gpu=0)

                # Add flwr client code
                for file in files:
                    job.to(str(file.absolute()), f"site-{i}")

            # TODO: Use job export and simulator run
            job.export_job("/Users/panheng/Projects/NVFlare/.cache/jobs/job_config")
            job_path = Path("/Users/panheng/Projects/NVFlare/.cache/jobs/job_config") / job_name
            print(f"Job exported to: {str(job_path)}")
            # job.simulator_run("/Users/panheng/Projects/NVFlare/.cache/jobs/workdir")

            self.sess.submit_job(str(job_path))

            # TODO: Return RunTracker
            return RunTracker(run_id=0, proc=None)  # Replace with actual run_id and proc
        except Exception as e:
            # TODO: log error
            raise e from None

    @property
    def sess(self) -> Session:
        if self._sess is None:
            self._sess = new_secure_session(
                "admin@nvidia.com", "/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com"
            )
            print("Session created.")
        return self._sess


executor = FlareEngine()
