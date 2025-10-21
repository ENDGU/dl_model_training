
# ML Project Template

This is a PyTorch Lightning project template that provides a clean and flexible structure for training machine learning models. It supports tasks like image classification, tabular data classification, and easy model deployment.

## Project Structure

```
ml-project/
├─ README.md                      # Project documentation
├─ requirements.txt               # Python dependencies
├─ pyproject.toml                 # Project metadata and configuration
├─ configs/                       # Configuration files
│  ├─ default.yaml                # Default configuration
│  ├─ image_classification.yaml   # Example: Image classification configuration
│  └─ tabular_classification.yaml # Example: Tabular data classification configuration
├─ data/                          # Raw, processed, and interim data
│  ├─ raw/                        # Original raw data
│  ├─ interim/                    # Intermediate datasets
│  └─ processed/                  # Final processed datasets for training/validation/testing
├─ outputs/                       # Logs, weights, model checkpoints, and evaluation metrics
├─ src/                           # Source code for the project
│  ├─ models/                     # Model definitions (e.g., backbone, architecture)
│  ├─ data/                       # Data processing and dataset definitions
│  ├─ utils/                      # Utility functions (e.g., seed setting, logging)
│  ├─ train.py                    # Training script
│  ├─ evaluate.py                 # Evaluation script
│  ├─ test.py                     # Testing script
│  └─ predict.py                  # Prediction and export script (e.g., ONNX)
└─ tests/                         # Unit tests
   └─ test_smoke.py               # Smoke tests to check if the project runs
```

## Requirements

You can install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Raw Data**:
   Your raw data should be stored in the `data/raw/` folder. It can be in any format (e.g., images, CSVs).

2. **Prepare the Data**:
   Use the provided script `prepare_dataset.py` to process your raw data and create splits for training, validation, and testing.

   ```bash
   python src/prepare_dataset.py --raw_dir data/raw --out_dir data/processed --val_ratio 0.1 --test_ratio 0.1
   ```

   This script:
   - Takes images from `data/raw/` and generates CSV files (`train.csv`, `val.csv`, `test.csv`) in `data/processed/`.
   - Creates a `class_index.json` mapping for the class names.

3. **Dataset Format**:
   For image classification, each row in the CSV should have:
   - `path`: Full path to the image file.
   - `label`: The label associated with the image (an integer).

## Configuration

All configuration settings are stored in YAML files located in the `configs/` folder.

- The **default.yaml** file contains default settings for the project.
- You can create your own configuration for different tasks (e.g., image classification, tabular data classification).

Example of changing configuration for a custom dataset:

```yaml
dataset:
  name: "ImageFolderCSV"         # Dataset type
  train_csv: "train.csv"         # Training CSV
  val_csv: "val.csv"             # Validation CSV
  test_csv: "test.csv"           # Test CSV
  input_col: "path"              # Column name for image file paths
  label_col: "label"             # Column name for image labels
  class_map_path: "class_index.json" # Path to class index file
```

### Available Configurations

- **image_classification.yaml**: For training an image classification model.
- **tabular_classification.yaml**: For training a tabular data classification model.

## Training the Model

1. **Training**:
   To start training, run the `train.py` script:

   ```bash
   python src/train.py -c configs/image_classification.yaml
   ```

   This will:
   - Load the dataset as specified in the configuration file.
   - Train the model with the specified architecture and hyperparameters.
   - Save the model checkpoints and logs to the `outputs/` directory.

   Hyperparameters such as batch size, learning rate, and the number of epochs can be configured in the YAML files.

2. **Model Checkpoints**:
   The best model (based on validation accuracy) will be saved to the `outputs/checkpoints/` directory.

## Evaluation

To evaluate the model, run the following script:

```bash
python src/evaluate.py --config configs/image_classification.yaml --ckpt outputs/checkpoints/last.ckpt
```

This script will:
- Load the model from the specified checkpoint.
- Evaluate the model on the validation and test datasets.

## Testing

To test the model on the test set:

```bash
python src/test.py --config configs/image_classification.yaml --ckpt outputs/checkpoints/last.ckpt
```

This will:
- Load the best model checkpoint and evaluate it on the test set.

## Prediction and Export

1. **Making Predictions**:
   Use the `predict.py` script to make predictions on new images.

   ```bash
   python src/predict.py --config configs/image_classification.yaml --ckpt outputs/checkpoints/last.ckpt --image /path/to/image.jpg
   ```

   This will:
   - Load the model from the specified checkpoint.
   - Make a prediction for the input image.

2. **Exporting Model**:
   To export the model as an ONNX file:

   ```bash
   python src/predict.py --config configs/image_classification.yaml --ckpt outputs/checkpoints/last.ckpt --image /path/to/image.jpg --export_onnx model.onnx
   ```

   This will export the model to an ONNX format, which can be used for inference in other platforms.

## Logging and Checkpoints

- The project uses **TensorBoard** logging via PyTorch Lightning. The logs will be saved in the `outputs/` directory.
- **Model checkpoints** are automatically saved during training, and the best model is saved based on validation accuracy.

## Common Issues and Solutions

1. **Memory Issues (OOM)**
   If you run into memory issues, try the following:
   - Reduce the batch size in the configuration file (`batch_size`).
   - Use `precision: 16` in the configuration to enable mixed precision training.

2. **CUDA Not Available**
   If your machine does not have a GPU or CUDA installed, you can change the training device to CPU by setting `accelerator: "cpu"` in the config.

3. **DataLoader Deadlock**
   On Windows, the DataLoader can sometimes get stuck if not properly set up. Ensure that your script is protected by `if __name__ == "__main__":`.

4. **Class Mismatch**
   Ensure that the `class_index.json` file is properly loaded. It must match the class labels used during training and prediction.

## Contributing

Feel free to fork this repository and create your own custom models or improvements. If you find a bug or have a feature request, please open an issue.

## TODO
