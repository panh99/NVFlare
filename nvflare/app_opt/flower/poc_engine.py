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
from logging import ERROR
from pathlib import Path
from typing import List

import pathspec
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.common.config import get_project_config
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.superexec.executor import Executor as FlowerSuperExecExecutor
from flwr.superexec.executor import RunTracker
from typing_extensions import Dict, Optional, override

import nvflare.tool.poc.poc_commands as cmds
from nvflare import FedJob
from nvflare.app_opt.flower.controller import FlowerController
from nvflare.app_opt.flower.executor import FlowerExecutor
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session

# Directory where jobs will be exported
# DEFAULT_JOB_DIR = "/tmp/nvflare/flower_job_config"
DEFAULT_JOBS_DIR = "/home/pan/NVFlare/.cache"


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


class PocEngine(FlowerSuperExecExecutor):
    """POC engine executor for Flower SuperExec."""

    def __init__(self) -> None:
        self._sess = None
        self.job_dir = DEFAULT_JOBS_DIR
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
            fab_version, fab_id = get_fab_metadata(fab_file)

            # Install FAB
            fab_path = install_from_fab(fab_file, None, True)

            # Retrieve the FAB info
            config = get_project_config(fab_path)
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

            # TODO: send override_config to Server Job and then to the SuperLink
            # Define the controller workflow and send to server
            controller = FlowerController(server_app=server_app_ref)
            job.to(controller, "server")

            # Add flwr server code
            for file in files:
                job.to(str(file.absolute()), "server")

            # Add clients
            for i in range(1, n_clients + 1):
                executor = FlowerExecutor(client_app=client_app_ref)
                job.to(executor, f"site-{i}", gpu=0)

                # Add flwr client code
                for file in files:
                    job.to(str(file.absolute()), f"site-{i}")

            # Export job
            job_path = Path(self.job_dir) / job_name
            job.export_job(job_path.parent)

            # TODO: Remove this line
            print(f"Job exported to: {str(job_path)}")

            self.sess.submit_job(str(job_path))

            # TODO: Return RunTracker
            return RunTracker(run_id=0, proc=None)  # Replace with actual run_id and proc
        except Exception as e:
            import traceback

            log(ERROR, "Could not start run: %s", traceback.format_exc())
            return None

    @property
    def sess(self) -> Session:
        if self._sess is not None:
            return self._sess

        # Obtain the POC workspace
        workspace = cmds.get_poc_workspace()

        # Load the project config and the service config
        project_config, service_config = cmds.setup_service_config(workspace)

        # Obtain the admin user name
        admin_username: str = cmds.get_proj_admin(project_config)

        # Obtain the path to start.sh
        shell_path = cmds.get_service_command(
            cmd_type=cmds.CMD_START_POC,
            prod_dir=cmds.get_prod_dir(workspace),
            service_dir=admin_username,
            service_config=service_config,
        )

        # Create a new session
        startup_kit_path = str(Path(shell_path).parent.parent)
        self._sess = new_secure_session(username=admin_username, startup_kit_location=startup_kit_path)

        # TODO: Remove this line
        print(f"Session created:\nusername={admin_username}\nstartup_kit_location={startup_kit_path}")
        return self._sess


executor = PocEngine()
