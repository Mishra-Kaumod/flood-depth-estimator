"""
Utility to load configuration from `utils/config.cfg` and optionally export to environment variables.

Usage:
    from utils.config import load_config, get_config
    cfg = load_config()  # loads utils/config.cfg
    # or export to environment variables:
    cfg = load_config(export_env=True)

This helper prefers values from environment variables when present, and falls back to the config file.
Do NOT store production credentials in the repo; prefer environment variables or a secrets manager.
"""

import os
from configparser import ConfigParser
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / 'config.cfg'


def load_config(path: str = None, export_env: bool = False):
    """Load configuration from INI file and return a nested dict.

    If export_env is True, relevant keys will be set as environment variables
    (without overwriting existing env vars).
    """
    cfg_path = Path(path) if path else DEFAULT_PATH
    parser = ConfigParser()
    read_files = parser.read(cfg_path)
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = {}
    for section in parser.sections():
        cfg[section] = {}
        for key, val in parser.items(section):
            cfg[section][key] = val

    # export to environment variables if requested
    if export_env:
        # AWS
        aws = cfg.get('aws', {})
        _set_env_if_missing('AWS_ACCESS_KEY_ID', aws.get('aws_access_key_id'))
        _set_env_if_missing('AWS_SECRET_ACCESS_KEY', aws.get('aws_secret_access_key'))
        _set_env_if_missing('AWS_DEFAULT_REGION', aws.get('aws_region'))
        _set_env_if_missing('S3_BUCKET', aws.get('s3_bucket'))
        _set_env_if_missing('S3_PREFIX', aws.get('s3_prefix'))

        # App
        appc = cfg.get('app', {})
        _set_env_if_missing('FLASK_SECRET', appc.get('flask_secret'))
        _set_env_if_missing('OUTPUT_PREFIX', appc.get('output_prefix'))
        _set_env_if_missing('ANNOTATED_PREFIX', appc.get('annotated_prefix'))
        _set_env_if_missing('RESULTS_PREFIX', appc.get('results_prefix'))
        _set_env_if_missing('OBJECTS_PREFIX', appc.get('objects_prefix'))

        # Local
        local = cfg.get('local', {})
        _set_env_if_missing('LOCAL_OUTPUT_DIR', local.get('output_dir'))

    return cfg


def _set_env_if_missing(key: str, value: str):
    if not value:
        return
    if key in os.environ and os.environ.get(key):
        return
    os.environ[key] = value


def get_config():
    """Convenience: load config but prefer environment variables when present."""
    cfg = load_config()
    # prefer env vars
    aws = cfg.get('aws', {})
    aws['aws_access_key_id'] = os.getenv('AWS_ACCESS_KEY_ID', aws.get('aws_access_key_id'))
    aws['aws_secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY', aws.get('aws_secret_access_key'))
    aws['aws_region'] = os.getenv('AWS_DEFAULT_REGION', aws.get('aws_region'))
    aws['s3_bucket'] = os.getenv('S3_BUCKET', aws.get('s3_bucket'))
    aws['s3_prefix'] = os.getenv('S3_PREFIX', aws.get('s3_prefix'))

    appc = cfg.get('app', {})
    appc['flask_secret'] = os.getenv('FLASK_SECRET', appc.get('flask_secret'))

    local = cfg.get('local', {})
    local['output_dir'] = os.getenv('LOCAL_OUTPUT_DIR', local.get('output_dir'))

    return {
        'aws': aws,
        'app': appc,
        'local': local,
    }

