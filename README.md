## 📋 README: Amazon Price Prediction Challenge (2025)

This repository contains the code and resources for the **Amazon ML Challenge 2025** submission by **Team letsdoit**. The project focuses on predicting the price of Amazon products based on their catalog content and associated images.

---

## 🚀 Project Overview

The core of this project is a **Deep Learning Regression Model** built using TensorFlow and Keras. We employ a **multimodal approach** where product images (visual data) are used to predict the product price (numerical target).

A key step in the process was addressing the **highly skewed distribution** of the product prices. To achieve a more robust and accurate model, we trained the final network to predict the **logarithm of the price** ($\log(1+\text{price})$), which normalized the target variable distribution. The final predictions are then exponentiated back to the original price scale.

### Key Technologies

* **Python:** Programming Language
* **TensorFlow/Keras:** Deep Learning Framework
* [cite_start]**EfficientNetB0 (Pre-trained on ImageNet):** Base model for visual feature extraction [cite: 460, 463, 1395, 1397]
* **Pandas/Numpy:** Data manipulation and numerical operations
* [cite_start]**Scikit-learn:** Data splitting (e.g., `train_test_split`) [cite: 211, 406]
* **Matplotlib/Seaborn:** Data Visualization

---

## 🛠️ Data Preprocessing and Feature Engineering

[cite_start]The solution was developed using a subset of the full training data (the **middle 50,000 rows** which, after filtering, resulted in a clean set of **19,852 valid images** [cite: 41, 64, 422]).

### 1. Data Cleaning
* [cite_start]**Initial Slicing:** Loaded the full training data and selected the tail 50,000 rows (`df.tail(50000)`). [cite: 40, 64]
* [cite_start]**Image Path Creation:** Constructed local image paths by extracting the filename from the `image_link` URL and joining it with the image directory. [cite: 69, 780]
* **Corrupt Image Removal:** Implemented a rigorous two-stage filtering process:
    1.  [cite_start]Verified that the image file existed on disk. [cite: 71, 880]
    2.  [cite_start]Verified that the image file could be successfully decoded by TensorFlow to check for corruption. [cite: 407, 410, 770]
* [cite_start]**Resulting Clean Data:** A final dataset of **19,852** samples was used for training after removing non-existent or corrupted images. [cite: 422, 436]

### 2. Feature Engineering (for potential use with non-image models or future enhancements)
Numerical features were engineered from the `catalog_content` text:
* **`pack_count`:** Extracted from patterns like `(Pack of \d+)`. [cite_start]Defaults to 1. [cite: 368]
* [cite_start]**`item_size`:** Extracted the size in Ounce/oz/OZ. [cite: 369]
* [cite_start]**`total_size`:** Calculated as `pack_count * item_size`. [cite: 371]
* [cite_start]**`unit_price`:** Calculated as `price / (total_size + \epsilon)`. [cite: 372]
* [cite_start]**`brand`:** Extracted a multi-word brand name by identifying consecutive capitalized words at the start of the "Item Name." [cite: 349, 361, 364]
* [cite_start]**Target Transformation (`log_price`):** Applied the $\log(1+x)$ transformation to the raw `price` column to normalize its distribution. [cite: 340, 382]

### 3. Data Splitting
The clean dataset was split into training and validation sets using a **stratified split** to ensure that the distribution of prices was maintained across both sets:
* [cite_start]**Split Ratio:** $80\%$ Training, $20\%$ Validation [cite: 221, 389]
* [cite_start]**Stratification Key:** **`price_bin`**, created by dividing the raw price into 10 quantiles. [cite: 213, 385]
* **Final Shapes:**
    * [cite_start]Training Set: **15,881** rows [cite: 575]
    * [cite_start]Validation Set: **3,971** rows [cite: 576]

---

## 🧠 Model Architecture and Training

### Model
A **Transfer Learning** approach was used for the image feature extraction.

* [cite_start]**Base Model:** Pre-trained **EfficientNetB0** (on ImageNet)[cite: 460, 463]. [cite_start]The base model layers were **frozen** to prevent re-training on the limited dataset. [cite: 465, 1399]
* **Custom Head:** A sequence of layers added on top of the base model:
    1.  [cite_start]`GlobalAveragePooling2D` [cite: 469]
    2.  [cite_start]`Dense` layer with 128 units and ReLU activation [cite: 470, 1403]
    3.  [cite_start]`Dropout` layer (0.3 rate) [cite: 471, 1404]
    4.  [cite_start]Final `Dense` output layer with 1 unit (for regression) [cite: 472, 1405]

### Training Configuration
* **Initial Model (Raw Price Training - $\text{price}$):**
    * [cite_start]**Loss:** Mean Squared Error (`mean_squared_error`) [cite: 482]
    * [cite_start]**Metric:** Mean Absolute Error (`mean_absolute_error`) [cite: 483]
    * [cite_start]**Final Validation $\text{MAE}$:** Approximately **\$19.14** (from Epoch 4, before early stopping restored best weights). [cite: 532]

* **Final Model (Log-Transformed Price Training - $\log\_price$):**
    * [cite_start]**Target Variable:** `log_price` [cite: 1217, 1222]
    * [cite_start]**Loss:** Mean Squared Error (`mean_squared_error`) [cite: 1412]
    * [cite_start]**Optimizer:** Adam [cite: 481, 1411]
    * [cite_start]**Callbacks:** **Early Stopping** on `val_loss` with a `patience=3` to prevent overfitting. [cite: 495, 497, 1415, 1418]

---

## 💾 Submission and Artifacts

The final, best-performing model trained on the log-transformed price was used to generate predictions on the test set.

### Prediction Process
1.  [cite_start]**Test Data Loading:** Loaded the `test.csv` data and created image paths using the **`sample_id`** (corrected from the initial failure). [cite: 871, 875]
2.  **Valid Image Filtering:** Filtered the test set to include only samples with valid, existing images. [cite_start]**65,681** valid images were found. [cite: 881, 915]
3.  [cite_start]**Prediction:** The model predicted the **$\log\_price$** for all valid samples. [cite: 897]
4.  **Inverse Transformation:** The predicted $\log\_price$ values must be converted back to the raw price scale using the formula $\text{price} = e^{\log\_price} - 1$ before submission.
5.  **Submission File Creation:** Predicted prices were merged back into the full test set using `sample_id`. Missing prices (from invalid images) were imputed using the **mean predicted price** ($\approx \$24.5$ based on sample outputs). [cite_start]All final prices were ensured to be non-negative. [cite: 814, 818, 819, 905, 907]

### Key Artifacts Saved
| File Name | Description | Size |
| :--- | :--- | :--- |
| `amazon_price_log_model.keras` | [cite_start]**Final trained Keras model** (predicts $\log(\text{price})$). [cite: 1430] | [cite_start]~18.14 MB [cite: 1182] |
| `train_df_final.pkl` | [cite_start]Final feature-engineered training DataFrame. [cite: 1172] | [cite_start]17.28 MB [cite: 1182] |
| `val_df_final.pkl` | [cite_start]Final feature-engineered validation DataFrame. [cite: 1174] | [cite_start]4.39 MB [cite: 1182] |
| `test_out.csv` | [cite_start]**Final submission file** (75,000 rows). [cite: 1176] | [cite_start]1.81 MB [cite: 1182] |

[cite_start]The submission file contains **75,000 rows**, the required number for the challenge. [cite: 1195, 1196]

---

## 👩‍💻 How to Run (Local Setup)

The code snippets demonstrate a workflow typically executed in a **Google Colab** environment, relying on **Google Drive** mounts for data and model persistence.

### Prerequisites
1.  Python environment with the following libraries: `pandas`, `numpy`, `tensorflow`, `tqdm`, `scikit-learn`, `Pillow`, `matplotlib`, `seaborn`.
2.  All required Amazon dataset files (`train.csv`, `test.csv`, image folders) must be accessible at the specified Google Drive paths (e.g., `/content/drive/MyDrive/amazon_dataset/...`).

### Core Steps

1.  [cite_start]**Mount Google Drive:** If running in Colab, execute the mount command. [cite: 37]
2.  **Run Preprocessing/Training:** Execute the code for data cleaning, feature engineering, and the final model training using `model_log`.
3.  **Run Prediction Script:** Execute the final prediction script (corrected version on Page 21) to generate `test_out.csv`.

---

*Contact: Team letsdoit, Amazon ML Challenge 2025*
