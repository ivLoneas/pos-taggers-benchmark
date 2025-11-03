import os
import json
from pathlib import Path


def find_metrics_files(root_dir="."):
    """
    Search for metrics.json files in subdirectories.

    Args:
        root_dir: Root directory to start search (default: current directory)

    Returns:
        List of tuples (directory_name, metrics_data)
    """
    metrics_list = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "metrics.json" in filenames:
            metrics_path = os.path.join(dirpath, "metrics.json")
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Get the directory name (model name)
                    model_name = os.path.basename(dirpath)
                    metrics_list.append((model_name, data))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {metrics_path}: {e}")

    return metrics_list


def extract_metrics(data):
    """
    Extract relevant metrics from the JSON data.

    Args:
        data: Parsed JSON data

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
        'adj_noun_f1': None
    }

    # Extract global metrics
    if 'global' in data:
        global_data = data['global']
        metrics['accuracy'] = global_data.get('accuracy')
        metrics['precision'] = global_data.get('precision_weighted')
        metrics['recall'] = global_data.get('recall_weighted')
        metrics['f1'] = global_data.get('f1_weighted')

    # Extract ADJ+NOUN F1 from pairs
    if 'pairs' in data and 'ADJ+NOUN' in data['pairs']:
        metrics['adj_noun_f1'] = data['pairs']['ADJ+NOUN'].get('f1')

    return metrics


def generate_markdown_table(metrics_list):
    """
    Generate a markdown table from the metrics list.

    Args:
        metrics_list: List of tuples (model_name, metrics_data)

    Returns:
        String containing the markdown table
    """
    # Table header
    table = "| Model | Size | Accuracy | Precision | Recall | F1 | ADJ+NOUN F1 |\n"
    table += "|-------|------|----------|-----------|--------|----|--------------|\n"

    # Table rows
    for model_name, data in metrics_list:
        metrics = extract_metrics(data)

        # Format numbers to 4 decimal places, or show '-' if None
        accuracy = f"{metrics['accuracy']:.4f}" if metrics['accuracy'] is not None else '-'
        precision = f"{metrics['precision']:.4f}" if metrics['precision'] is not None else '-'
        recall = f"{metrics['recall']:.4f}" if metrics['recall'] is not None else '-'
        f1 = f"{metrics['f1']:.4f}" if metrics['f1'] is not None else '-'
        adj_noun_f1 = f"{metrics['adj_noun_f1']:.4f}" if metrics['adj_noun_f1'] is not None else '-'

        table += f"| {model_name} | | {accuracy} | {precision} | {recall} | {f1} | {adj_noun_f1} |\n"

    return table


def main():
    """Main function to execute the script."""
    # Search for metrics.json files in current directory and subdirectories
    metrics_list = find_metrics_files()

    if not metrics_list:
        print("No metrics.json files found in subdirectories.")
        return

    # Generate and print the markdown table
    markdown_table = generate_markdown_table(metrics_list)
    print(markdown_table)

    # Optionally, save to a file
    with open('metrics_summary.md', 'w', encoding='utf-8') as f:
        f.write(markdown_table)
    print("\nTable saved to 'metrics_summary.md'")


if __name__ == "__main__":
    main()