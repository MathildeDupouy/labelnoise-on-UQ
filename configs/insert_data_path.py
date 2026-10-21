# Script to insert the data paths into the config files
if __name__ == "__main__":
    import argparse
    import json
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Insert data paths into config files.")
    parser.add_argument("--config_dir", type=str, required=True, help="Directory containing the config files.")
    parser.add_argument("--data_dir", type=str, required=False, help="Directory containing the data files.")
    parser.add_argument("-r", "--reset", action="store_true", help="Reset data paths to anonymous paths.")
    args = parser.parse_args()
    
    reset = args.reset
    
    config_dir = Path(args.config_dir)
    assert config_dir.is_dir(), f"{config_dir} is not a valid directory."
    if not reset:
        assert args.data_dir is not None, "Data directory must be provided unless resetting."
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
            if not reset:
                config["dataset"]["csv path"] = str(Path(config["dataset"]["csv path"].replace("path/to", str(data_dir))))
            else :
                initial_path = Path(config["dataset"]["csv path"])
                print(f"Initial path: {initial_path}", initial_path.parts)
                parts = initial_path.parts
                identified_path = Path(*parts[:parts.index('data')])
                config["dataset"]["csv path"] = str(initial_path).replace(str(identified_path), "path/to")
        elif "data path" in config["dataset"].keys(): # MNIST
            if not reset:
                config["dataset"]["data path"] = str(Path(config["dataset"]["data path"].replace("path/to", str(data_dir))))
            else:
                initial_path = Path(config["dataset"]["data path"])
                print(f"Initial path: {initial_path}", initial_path.parts)
                parts = initial_path.parts
                identified_path = Path(*parts[:parts.index('data')])
                config["dataset"]["data path"] = str(initial_path).replace(str(identified_path), "path/to")
        elif "normal path" in config["dataset"].keys() or "abnormal path" in config["dataset"].keys(): # PTB
            if not reset:
                config["dataset"]["normal path"] = str(Path(config["dataset"]["normal path"].replace("path/to", str(data_dir))))
                config["dataset"]["abnormal path"] = str(Path(config["dataset"]["abnormal path"].replace("path/to", str(data_dir))))
            else:
                initial_normal_path = Path(config["dataset"]["normal path"])
                print(f"Initial normal path: {initial_normal_path}", initial_normal_path.parts)
                parts = initial_normal_path.parts
                identified_path = Path(*parts[:parts.index('data')])
                config["dataset"]["normal path"] = str(initial_normal_path).replace(str(identified_path), "path/to")
                
                initial_abnormal_path = Path(config["dataset"]["abnormal path"])
                print(f"Initial abnormal path: {initial_abnormal_path}", initial_abnormal_path.parts)
                parts = initial_abnormal_path.parts
                identified_path = Path(*parts[:parts.index('data')])
                config["dataset"]["abnormal path"] = str(initial_abnormal_path).replace(str(identified_path), "path/to")
        else:
            print(f"No recognized data path keys found in {config_file}. Skipping.")
            
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        if not reset:
            print(f"Updated {config_file} with data path {data_dir}.")
        else:
            print(f"Reset {config_file} to anonymous data paths.")