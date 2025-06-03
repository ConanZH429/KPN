import wandb
from .logger import Logger
from typing import Optional, Union
from pathlib import Path

class WandbLogger(Logger):
    def __init__(
            self,
            dir: str,
            project_name: str,
            name: str,
            config: dict,
    ):
        super().__init__()
        # login
        login = wandb.login(
            key="362451819636a5a2c9562546ebeab71c38fe6c7f",
            verify=True,
        )
        # init
        if login:
            self.run = wandb.init(
                project=project_name,
                dir=dir,
                name=name,
                config=config,
                force=True
            )
        self.text_num = 0
    
    def log_dict(self, data: dict, epoch: int):
        data["epoch"] = epoch
        self.run.log(
            data,
        )
    
    def log_text(self, text: str, step: Optional[int] = None):
        self.text_num += 1
        self.run.log(
            {
                f"text_{self.text_num}": text,
            }
        )
    
    def log_file(self, file_path: str, file_name: Optional[str] = None):
        self.run.log_artifact(
            file_path, file_name
        )

    def log_code(self, root_dir: str):
        self.run.log_code(root_dir)
    
    
    
