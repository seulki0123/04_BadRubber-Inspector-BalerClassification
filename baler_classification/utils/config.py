import yaml

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["baler_classification"]
    return config