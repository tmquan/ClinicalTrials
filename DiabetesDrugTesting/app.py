from lightning import CloudCompute, LightningApp, LightningFlow, LightningWork
from lightning.app.frontend import StreamlitFrontend
from lightning.app.storage import Path
from lightning.app.structures import Dict
import logging
import os
import re
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


class JupyterLabWork(LightningWork):
    def __init__(self, cloud_compute: Optional[CloudCompute] = None):
        super().__init__(cloud_compute=cloud_compute, parallel=True)
        self.pid = None
        self.token = None
        self.exit_code = None
        self.storage = None

    def run(self):
        self.storage = Path(".")

        jupyter_notebook_config_path = Path.home() / ".jupyter/jupyter_notebook_config.py"

        if os.path.exists(jupyter_notebook_config_path):
            os.remove(jupyter_notebook_config_path)

        with subprocess.Popen(
            f"{sys.executable} -m notebook --generate-config".split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            close_fds=True,
        ) as proc:
            self.pid = proc.pid

            self.exit_code = proc.wait()
            if self.exit_code != 0:
                raise Exception(self.exit_code)

        with open(jupyter_notebook_config_path, "a") as f:
            f.write(
                """c.NotebookApp.tornado_settings = {'headers': {'Content-Security-Policy': "frame-ancestors * 'self' "}}"""  # noqa E501
            )

        with open(f"jupyter_lab_{self.port}", "w") as f:
            proc = subprocess.Popen(
                f"{sys.executable} -m jupyter lab --ip {self.host} --port {self.port} --no-browser --config={jupyter_notebook_config_path}".split(" "),
                bufsize=0,
                close_fds=True,
                stdout=f,
                stderr=f,
            )

        with open(f"jupyter_lab_{self.port}") as f:
            while True:
                for line in f.readlines():
                    if "lab?token=" in line:
                        self.token = line.split("lab?token=")[-1]
                        proc.wait()

    @property
    def url(self):
        if not self.token:
            return ""
        if self._future_url:
            return f"{self._future_url}/lab?token={self.token}"
        else:
            return f"http://{self.host}:{self.port}/lab?token={self.token}"

class JupyterLabManager(LightningFlow):

    def __init__(self):
        super().__init__()
        self.jupyter_works = Dict()
        self.jupyter_configs = []

    def run(self):
        for idx, jupyter_config in enumerate(self.jupyter_configs):

            # Step 1: Create a new JupyterWork if a user requested it from the UI.
            username = jupyter_config["username"]
            if username not in self.jupyter_works:
                jupyter_config["ready"] = False

                # User can select GPU or CPU.
                cloud_compute = CloudCompute("gpu" if jupyter_config["use_gpu"] else "default")

                # HERE: We are creating the work dynamically !
                self.jupyter_works[username] = JupyterLabWork(cloud_compute=cloud_compute)

            # Step 2: Run the JupyterWork
            self.jupyter_works[username].run()

            # Step 3: Store the notebook token in the associated config.
            if self.jupyter_works[username].token:
                jupyter_config["token"] = self.jupyter_works[username].token

            # Step 4: Stop the work if the user requested it.
            if jupyter_config['stop']:
                self.jupyter_works[username].stop()
                self.jupyter_configs.pop(idx)

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)

def render_fn(state):
    import streamlit as st

    # Step 1: Enable users to select their notebooks and create them
    col1, col2, col3 = st.columns(3)
    with col1:
        create_jupyter = st.button("Create Jupyter Notebook")
    with col2:
        username = st.text_input('Enter your name', "tchaton")
        assert username
    with col3:
        use_gpu = st.checkbox('Use GPU')

    # Step 2: If a user clicked the button, add an element to the list of configs
    # Note: state.jupyter_configs = ... will sent the state update to the component.
    if create_jupyter:
        # Make username url friendly
        username = re.sub("[^0-9a-zA-Z]+", "_", username)
        new_config = [{"use_gpu": use_gpu, "token": None, "username": username, "stop": False}]
        state.jupyter_configs = state.jupyter_configs + new_config

    # Step 3: List of running notebooks.
    for idx, config in enumerate(state.jupyter_configs):
        col1, col2, col3 = st.columns(3)
        with col1:
            if not idx:
                st.write(f"Username")
            st.write(config['username'])
        with col2:
            if not idx:
                st.write(f"Use GPU")
            st.write(config['use_gpu'])
        with col3:
            if not idx:
                st.write(f"Stop")
            if config["token"]:
                should_stop = st.button("Stop this notebook", key=str(idx))

                # Step 4: Change stop if the user clicked the button
                if should_stop:
                    config["stop"] = should_stop
                    state.jupyter_configs = state.jupyter_configs

class RootFlow(LightningFlow):

    def __init__(self):
        super().__init__()
        self.manager = JupyterLabManager()

    def run(self):
        self.manager.run()

    def configure_layout(self):
        layout = [{"name": "Manager", "content": self.manager}]
        for config in self.manager.jupyter_configs:
            if not config['stop']:
                username = config['username']
                jupyter_work = self.manager.jupyter_works[username]
                layout.append(
                    {"name": f"JupyterLab {username}", "content": jupyter_work}
                )
        return layout


app = LightningApp(RootFlow())