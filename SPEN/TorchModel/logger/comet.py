import comet_ml
from .logger import Logger
from typing import Optional, Union
from pathlib import Path

class CometLogger(Logger):
    def __init__(self,
                 api_key: Optional[str] = None,
                 workspace: Optional[str] = None,
                 project_name: str = "Project",
                 experiment_name: str = "Experiment",
                 mode: Optional[str] = "create",
                 online: Optional[bool] = True
                 ):
        """
        Initialize the CometLogger class

        Args:
            api_key (Optional[str], optional): The Comet API key. Defaults to None.
            workspace (Optional[str], optional): The Comet workspace. Defaults to None.
            project_name str: The Comet project name. Defaults to None.
            experiment_name str: The Comet experiment name. Defaults to None.
            mode (Optional[str], optional): The Comet mode. Defaults to "create".
            online (Optional[bool], optional): Whether to log online. Defaults to True.
        """
        super().__init__()
        ExpConfig = comet_ml.ExperimentConfig(
            parse_args=False,
            display_summary_level=0,

        )
        self.experiment = comet_ml.start(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            mode=mode,
            online=online,
            experiment_config=ExpConfig,
        )
        self.experiment.set_name(experiment_name)

    def log_dict(self, data: dict, epoch: int):
        """
        Log a dictionary to Comet

        Args:
            data (dict): The dictionary to log
        """
        self.experiment.log_metrics(data, epoch=epoch)
    
    def log_text(self, text: str, step: Optional[int] = None):
        """
        Log a text to Comet

        Args:
            text (str): The text to log
        """
        self.experiment.log_text(text, step=step)
    
    def log_file(self, file_path: str, file_name: Optional[str] = None):
        """
        Log a file to Comet

        Args:
            file_path (str): The file path to log
            file_name (Optional[str], optional): The file name. Defaults to None.
        """
        self.experiment.log_asset(file_path, file_name=file_name ,overwrite=True)
    
    def log_code(self,
                 file_path: Optional[Union[str, Path]] = None,
                 folder_path: Optional[Union[str, Path]] = None,
                 code_name: Optional[str] = None,
                 overwrite: bool = False):
        """
        Log a code file to Comet

        Args:
            file_path (Optional[Union[str, Path]], optional): The file path. Defaults to None.
            folder_path (Optional[Union[str, Path]], optional): The folder path. Defaults to None.
            code_name (Optional[str], optional): The code name. Defaults to None.
            overwrite (bool, optional): Whether to overwrite. Defaults to False.
        """
        if file_path:
            self.experiment.log_code(file_name=str(file_path), code_name=code_name, overwrite=overwrite)
        elif folder_path:
            self.experiment.log_code(folder=str(folder_path), code_name=code_name, overwrite=overwrite)

    def log_hyperparams(self, hyperparams: dict):
        """
        Log hyperparameters to Comet

        Args:
            hyperparams (dict): The hyperparameters to log
        """
        self.experiment.log_parameters(hyperparams, nested_support=True)