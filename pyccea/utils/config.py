import toml

def load_params(conf_path):
    """
    Load parameters of a Cooperative Co-Evolutionary Algorithm (CCEA) from configuration file.

    Parameters
    ----------
    conf_path: str
        Path to the configuration file.
    Returns
    -------
    conf: dict
        Configuration parameters of a CCEA.
    """
    conf = toml.load(conf_path)
    return conf