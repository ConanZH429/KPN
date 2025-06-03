from typing import Optional

class Logger:
    def __init__(self):
        pass
    
    def log_dict(self, data: dict, epoch: int):
        """Log a dictionary of data to the logger"""
    
    def log_text(self, text: str):
        """Log a text to the logger"""
    
    def log_file(self, file_path: str):
        """Log a file to the logger"""
    
    def log_code(self,
                 file_path: Optional[str] = None,
                 folder_path: Optional[str] = None,
                 code_name: Optional[str] = None,
                 overwrite: bool = False):
        """Log a code file to the logger"""
    
    def log_hyperparams(self, hyperparams: dict):
        """Log hyperparameters to the logger"""