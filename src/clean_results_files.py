import json
import os
import glob
import csv
import shutil

PROJECT_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

def summarize_results_to_csv(results_dir, output_csv):
    """
    Parse the result JSON files and summarize the last values of test_dice, test_iou, test_specificity, test_sensitivity, and test_accuracy for every model in a CSV file.
    
    Args:
        results_dir (str): Directory containing the result JSON files.
        output_csv (str): Path to the output CSV file.
    """
    # Initialize a list to store the summary
    summary = []

    # Find all JSON files in the results directory
    json_files = glob.glob(os.path.join(results_dir, "*.json"))

    # Iterate over each JSON file
    for json_file in json_files:
        dataset = "drive" if "drive" in json_file else "ph2"
        with open(json_file, 'r') as f:
            data = json.load(f)
            for model_result in data:
                model = model_result['description']
                criterion_description = model_result.get('criterion', 'N/A')
                transforms = model_result.get('transform', 'N/A')
                test_dice = model_result['test_dice'][-1] if model_result['test_dice'] else None
                test_iou = model_result['test_iou'][-1] if model_result['test_iou'] else None
                test_specificity = model_result['test_specificity'][-1] if model_result['test_specificity'] else None
                test_sensitivity = model_result['test_sensitivity'][-1] if model_result['test_sensitivity'] else None
                test_accuracy = model_result['test_acc'][-1] if model_result['test_acc'] else None
                summary.append([model, dataset, criterion_description, transforms, test_dice, test_iou, test_specificity, test_sensitivity, test_accuracy])

    # Sort the summary first by dataset (ph2 first, then drive) and then by test_iou in descending order
    summary.sort(key=lambda x: (x[1] != 'ph2', -x[5] if x[5] is not None else float('-inf')))

    # Write the summary to a CSV file
    # Separate summaries for ph2 and drive datasets
    ph2_summary = [row for row in summary if row[1] == 'ph2']
    drive_summary = [row for row in summary if row[1] == 'drive']

    # Sort the summaries by test_iou in descending order
    ph2_summary.sort(key=lambda x: -x[5] if x[5] is not None else float('-inf'))
    drive_summary.sort(key=lambda x: -x[5] if x[5] is not None else float('-inf'))

    # Write the ph2 summary to a CSV file
    ph2_output_csv = output_csv.replace('.csv', '_ph2.csv')
    with open(ph2_output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Model', 'Dataset', 'Criterion Description', 'Transforms', 'Test Dice', 'Test IoU', 'Test Specificity', 'Test Sensitivity', 'Test Accuracy'])
        csvwriter.writerows(ph2_summary)

    # Write the drive summary to a CSV file
    drive_output_csv = output_csv.replace('.csv', '_drive.csv')
    with open(drive_output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Model', 'Dataset', 'Criterion Description', 'Transforms', 'Test Dice', 'Test IoU', 'Test Specificity', 'Test Sensitivity', 'Test Accuracy'])
        csvwriter.writerows(drive_summary)

    # Write the drive summary to a CSV file
    drive_output_csv = output_csv.replace('.csv', '_drive.csv')
    with open(drive_output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Model Name',  'Dataset', 'Criterion', 'Transforms', 'Test Dice', 'Test IoU', 'Test Specificity', 'Test Sensitivity', 'Test Accuracy'])
        csvwriter.writerows(drive_summary)

def clean_results_file(file_path):
    """Removes all contents of the result JSON file and leaves an empty list."""
    with open(file_path, 'w') as f:
        json.dump([], f, indent=4)

def archive_results_and_clean(results_dir):
    """
    Creates a new directory in results named experiment_x, copies the content of the JSON files and the summary file there,
    and then calls the clean_results_file function on the original JSON files.
    
    Args:
        results_dir (str): Directory containing the result JSON files.
    """
    # Create a new directory named experiment_x
    # Determine the next unused experiment number
    summary_csv_path = os.path.join(PROJECT_BASE_DIR, "results/summary.csv")
    summarize_results_to_csv(os.path.join(PROJECT_BASE_DIR, "results"), summary_csv_path)

    experiment_dirs = glob.glob(os.path.join(results_dir, "experiment_*"))
    experiment_numbers = [int(os.path.basename(d).split('_')[1]) for d in experiment_dirs if os.path.basename(d).split('_')[1].isdigit()]
    next_experiment_number = max(experiment_numbers, default=0) + 1
    experiment_dir = os.path.join(results_dir, f"experiment_{next_experiment_number}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Copy JSON files to the new directory
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    for json_file in json_files:
        shutil.copy(json_file, experiment_dir)

    # Copy the summary CSV files to the new directory
    ph2_summary_csv_path = summary_csv_path.replace('.csv', '_ph2.csv')
    drive_summary_csv_path = summary_csv_path.replace('.csv', '_drive.csv')
    shutil.copy(summary_csv_path, experiment_dir)
    shutil.copy(ph2_summary_csv_path, experiment_dir)
    shutil.copy(drive_summary_csv_path, experiment_dir)

    saved_model_dir = os.path.join(experiment_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    # Copy the models from the results/saved_models directory to the new saved_model directory
    saved_models_src_dir = os.path.join(results_dir, "saved_models")
    saved_model_files = glob.glob(os.path.join(saved_models_src_dir, "*.pth"))
    for model_file in saved_model_files:
        shutil.copy(model_file, saved_model_dir)

    # # Clean the original JSON files
    # for json_file in json_files:
    #     clean_results_file(json_file)

    # # Clean the original CSV files
    # csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    # for csv_file in csv_files:
    #     with open(csv_file, 'w', newline='') as f:
    #         f.truncate()
        

# Example usage
if __name__ == "__main__":
    archive_results_and_clean(os.path.join(PROJECT_BASE_DIR, "results"))