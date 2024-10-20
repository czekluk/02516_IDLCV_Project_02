import torch
import json
import os

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_results(outputs, path=os.path.join(PROJECT_BASE_DIR, "results/experiments.json")):
    """
    Saves the best model from outputs (the parameters).
    Saves all of the results into a results/experiments.json
    """
    saved_models_path = os.path.join(PROJECT_BASE_DIR, "results/saved_models")
    # save the best model
    best_model = outputs[0]
    torch.save(best_model["model"].state_dict(), 
               os.path.join(saved_models_path, 
                            f"{best_model['description']}-{best_model['timestamp'].year}-{best_model['timestamp'].month}-{best_model['timestamp'].day}_{best_model['timestamp'].hour}-{best_model['timestamp'].minute}-{best_model['timestamp'].second}-{best_model['test_acc'][-1]:.4f}-{best_model['model_name']}.pth"))
    
    # save the results
    data= []
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except:
            data = []

    # stringify the model and optimizer from the outputs as they are not json serializable
    for output in outputs:
        if "model" in output:
            output["model"] = str(output["model"])
        if "optimizer" in output["optimizer_config"]:
            output["optimizer_config"]["optimizer"] = str(output["optimizer_config"]["optimizer"])
        if "criterion" in output:
            output["criterion"] = str(output["criterion"])
        if "transform" in output:
            output["transform"] = str(output["transform"])
        if "timestamp" in output:
            output["timestamp"] = str(output["timestamp"])
    
    data.extend(outputs)
    # sort the entries
    data = sorted(data, key=lambda x: x['test_acc'][-1], reverse=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)