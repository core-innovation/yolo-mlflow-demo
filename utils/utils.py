import yaml
import logging

def load_yaml_config(
    config_path: str
) -> dict:
    """
    Loads a YAML configuration file.
    
    :param config_path: Path to the YAML file.
    :return: Parsed YAML file as a Python dictionary.
    """
    with open(config_path, 'r', encoding='utf8') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging():
    """
    Sets up the logging configuration.
    
    :return: Configured logger instance.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    return logger