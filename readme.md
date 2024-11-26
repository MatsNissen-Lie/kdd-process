# Data Science labs

## Table of Contents

<!-- 1. [Data Science Labs](#data-science-labs) -->

<!-- 2. [Access the Code for Analysis](#access-the-code-for-analysis)
3. [Datasets Used in the Project](#datasets-used-in-the-project)
   - [Classification Datasets](#classification-datasets)
   - [Forecasting Datasets](#forecasting-datasets)
4. [File Structure](#file-structure)
   - [Analysis](#analysis)
   - [Utilities](#utilities)
5. [Set Up LaTeX for Writing Reports](#set-up-latex-for-writing-reports)
   - [Step 1: Install TeX Live with Homebrew](#step-1-install-tex-live-with-homebrew)
   - [Step 2: Install the LaTeX Workshop Extension](#step-2-install-the-latex-workshop-extension)
   - [Step 3: Configure LaTeX Workshop](#step-3-configure-latex-workshop)
   - [Step 5: Compile Your LaTeX Document](#step-5-compile-your-latex-document) -->

- [Access the Code for Analysis](#access-the-code-for-analysis)
- [Datasets Used in the Project](#datasets-used-in-the-project)
  - [Classification Datasets](#classification-datasets)
  - [Forecasting Datasets](#forecasting-datasets)
- [File Structure](#file-structure)
  - [Analysis](#analysis)
  - [Utilities](#utilities)
- [Set Up LaTeX for Writing Reports](#set-up-latex-for-writing-reports)
  - [Step 1: Install TeX Live with Homebrew](#step-1-install-tex-live-with-homebrew)
  - [Step 2: Install the LaTeX Workshop Extension](#step-2-install-the-latex-workshop-extension)
  - [Step 3: Configure LaTeX Workshop](#step-3-configure-latex-workshop)
  - [Step 5: Compile Your LaTeX Document](#step-5-compile-your-latex-document)

## Access the Code for Analysis

All the code used for the analysis can be found [here](https://web.ist.utl.pt/~claudia.antunes/DSLabs/#).

This resource contains additional details and examples that complement the work in this project.

## Datasets Used in the Project

This project utilizes several datasets for classification and forecasting tasks. The datasets are organized based on their focus on either **economic** or **security** topics.

### Classification Datasets

- `c_e_class_financial_distress.csv`: A dataset focused on classifying financial distress (economic).
- `c_s_class_ny_arrests.csv`: A dataset focused on classifying arrests in New York (security).

### Forecasting Datasets

- `f_e_forecast_gdp_europe.csv`: A dataset for forecasting GDP trends in Europe (economic).
- `f_s_forecast_ny_arrests.csv`: A dataset for forecasting arrest trends in New York (security).

All datasets are stored in the `../data/` directory and are referenced accordingly in the project code.

## File Structure

The `src` directory is organized to align with the modular structure of the project, separating profiling tasks for classification and forecasting datasets. Below is an updated directory structure:

```plaintext
src/
├── module_1_profiling/
│ ├── classification_profiling/
│ │ ├── dimensionality.py
│ │ ├── distribution.py
│ │ ├── granularity.py
│ │ ├── sparsity.py
│ │ └── **init**.py
│ ├── forecasting_profiling/
│ │ ├── dimensionality.py
│ │ ├── distribution.py
│ │ ├── granularity.py
│ │ ├── sparsity.py
│ │ └── **init**.py
│ └── **init**.py
├── module_2_preparation/
│ ├── methodology.py
│ ├── encoding.py
│ ├── missing_values.py
│ ├── outliers.py
│ ├── scaling.py
│ ├── balancing.py
│ ├── discretization.py
│ └── **init**.py
├── module_3_feature_engineering/
│ ├── feature_selection.py
│ ├── feature_extraction.py
│ ├── feature_generation.py
│ └── **init**.py
├── module_4_classification/
│ ├── evaluation.py
│ ├── naive_bayes.py
│ ├── knn.py
│ ├── decision_trees.py
│ ├── logistic_regression.py
│ ├── neural_networks.py
│ ├── svm.py
│ ├── random_forests.py
│ ├── gradient_boosting.py
│ └── **init**.py
├── module_5_prediction/
│ ├── evaluation.py
│ ├── regression_trees.py
│ ├── logistic_regression.py
│ ├── neural_networks.py
│ ├── random_forests.py
│ ├── gradient_boosting.py
│ └── **init**.py
├── analysis/
│ ├── economic/
│ │ ├── classification_analysis.py
│ │ ├── forecasting_analysis.py
│ │ └── **init**.py
│ ├── security/
│ │ ├── classification_analysis.py
│ │ ├── forecasting_analysis.py
│ │ └── **init**.py
│ └── **init**.py
└── utils/
├── data_loader.py
├── preprocess.py
└── **init**.py
```

<!-- ### Modules -->

<!-- #### Module 1: Data Profiling

The `module_1_profiling/` folder is divided into subfolders for **classification** and **forecasting** profiling:

- **`classification_profiling/`**:
  - `dimensionality.py`: Analyze the dimensions of classification datasets.
  - `distribution.py`: Study data distributions for classification tasks.
  - `granularity.py`: Assess the granularity of classification datasets.
  - `sparsity.py`: Measure sparsity in classification datasets.
- **`forecasting_profiling/`**:
  - `dimensionality.py`: Analyze the dimensions of forecasting datasets.
  - `distribution.py`: Study data distributions for forecasting tasks.
  - `granularity.py`: Assess the granularity of forecasting datasets.
  - `sparsity.py`: Measure sparsity in forecasting datasets.

#### Module 2: Data Preparation

The `module_2_preparation/` folder includes functions for data preparation, such as handling missing values, outlier removal, and scaling.

#### Module 3: Feature Engineering

The `module_3_feature_engineering/` folder includes scripts for:

- Feature selection
- Feature extraction
- Feature generation

#### Module 4: Classification

The `module_4_classification/` folder provides algorithms and evaluation methods for classification tasks.

#### Module 5: Prediction

The `module_5_prediction/` folder provides algorithms and evaluation methods for forecasting tasks.

--- -->

### Analysis

The `analysis/` folder applies the modules to the datasets:

- **`economic/`**: Scripts for classification and forecasting analysis on economic datasets.
- **`security/`**: Scripts for classification and forecasting analysis on security datasets.

---

### Utilities

The `utils/` folder contains helper scripts:

<!-- - `data_loader.py`: Functions for loading datasets. -->

- `preprocess.py`: Common preprocessing utilities.

## Set Up LaTeX for Writing Reports

### Step 1: Install TeX Live with Homebrew

1. Open your terminal.

2. Run the following command:

   ```bash
   brew install --cask mactex-no-gui
   ```

   - **`mactex-no-gui`** installs TeX Live without the GUI applications (e.g., TeXShop), which is a lighter version (~3GB instead of ~4.4GB).
   - If you want the full MacTeX suite (with GUI tools), use:

     ```bash
     brew install --cask mactex
     ```

### Step 2: Install the LaTeX Workshop Extension

1. Open Visual Studio Code.
2. Navigate to the Extensions view by pressing `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (macOS).
3. Search for **LaTeX Workshop** and click **Install**.

The **LaTeX Workshop** extension provides features such as:

- Syntax highlighting
- Live preview
- PDF syncing

Here’s the formatted version for your README:

### Step 3: Configure LaTeX Workshop

LaTeX Workshop automatically builds the PDF and supports live preview by default. However, you may want to customize some settings:

1. Open the settings JSON in VS Code:

   - Press `Ctrl+,` (Windows/Linux) or `Cmd+,` (macOS), then search for "settings.json."

2. Add or modify the following settings:

   ```json
   {
     "latex-workshop.latex.autoBuild.run": "onSave",
     "latex-workshop.view.pdf.viewer": "tab"
   }
   ```

   - `"onSave"`: Automatically compiles the project whenever you save your `.tex` file.
   - `"tab"`: Displays the PDF in an internal VS Code tab. You can use `"browser"` instead to open the PDF in an external browser.

### Step 5: Compile Your LaTeX Document

1. Save your `.tex` file (`Ctrl+S` on Windows/Linux or `Cmd+S` on macOS).
2. LaTeX Workshop will automatically compile the document.
3. The PDF viewer will reload whenever a new version of the document is compiled.
