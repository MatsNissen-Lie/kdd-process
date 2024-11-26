# Data Science labs

## Access the Code for Analysis

All the code used for the analysis can be found [here](https://web.ist.utl.pt/~claudia.antunes/DSLabs/#).

This resource contains additional details and examples that complement the work in this project.

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

### Step 3: Configure LaTeX Workshop (Done)

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
