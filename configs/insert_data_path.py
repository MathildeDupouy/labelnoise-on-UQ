# Script to insert the data paths into the config files
if __name__ == "__main__":
    import argparse
    import json
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Insert data paths into config files.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the config files.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the data files.")
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    assert config_dir.is_dir(), f"{config_dir} is not a valid directory."
    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"{data_dir} is not a valid directory."
    assert (data_dir/"data").is_dir(), f"{data_dir} needs to contain a 'data' subdirectory."
    
    for config_file in config_dir.glob("**/*.json"):
        print(f"Processing {config_file}...")
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Insert the data path into the config
        if "dataset" not in config.keys():
            print(f"No 'dataset' key found in {config_file}. Skipping.")
            continue
        if "csv path" in config["dataset"].keys(): # ESR
            config["dataset"]["csv path"] = str(Path(config["dataset"]["csv path"].replace("path/to/", str(data_dir))))
        elif "data path" in config["dataset"].keys(): # MNIST
            config["dataset"]["data path"] = str(Path(config["dataset"]["data path"].replace("path/to/", str(data_dir))))
        elif "normal path" in config["dataset"].keys() or "abnormal path" in config["dataset"].keys(): # PTB
            config["dataset"]["normal path"] = str(Path(config["dataset"]["normal path"].replace("path/to/", str(data_dir))))
            config["dataset"]["abnormal path"] = str(Path(config["dataset"]["abnormal path"].replace("path/to/", str(data_dir))))
        else:
            print(f"No recognized data path keys found in {config_file}. Skipping.")
            
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"Updated {config_file} with data path {data_dir}.")