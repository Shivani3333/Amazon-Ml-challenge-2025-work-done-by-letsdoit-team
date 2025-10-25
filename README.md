## 📋 README: Amazon Price Prediction Challenge (2025)

[cite_start]This repository contains the code and resources for the **Amazon ML Challenge 2025** submission by **Team letsdoit**[cite: 98, 397]. [cite_start]The project focuses on predicting the price of Amazon products based on their catalog content and associated images[cite: 365, 450].

***

## 🚀 Project Overview

[cite_start]The core of this project is a **Deep Learning Regression Model** built using TensorFlow and Keras[cite: 270, 473]. [cite_start]We employ a **multimodal approach** where product images (visual data) are used to predict the product price (numerical target)[cite: 450, 454].

[cite_start]A key step in the process was addressing the **highly skewed distribution** of the product prices[cite: 294, 307]. [cite_start]To achieve a more robust and accurate model, we trained the final network to predict the **logarithm of the price** ($\log(1+\text{price})$), which normalized the target variable distribution[cite: 323, 339, 1225]. [cite_start]The final predictions are then exponentiated back to the original price scale[cite: 819, 907].

### Key Technologies

* **Python**
* [cite_start]**TensorFlow/Keras** [cite: 270, 459]
* [cite_start]**EfficientNetB0 (Pre-trained on ImageNet):** Base model for visual feature extraction [cite: 460, 463]
* [cite_start]**Pandas/Numpy** [cite: 3, 7]
* [cite_start]**Scikit-learn:** Used for data splitting (e.g., `train_test_split`) [cite: 211, 406]
* [cite_start]**Matplotlib/Seaborn:** Used for data visualization [cite: 55, 290]

***

## 🛠️ Data Preprocessing and Feature Engineering

[cite_start]The solution was developed using a subset of the full training data (the **middle 50,000 rows**) [cite: 40, 64][cite_start], which, after filtering, resulted in a clean set of **19,852 valid images**[cite: 422, 436].

### 1. Data Cleaning
* [cite_start]**Initial Slicing:** Loaded the full training data and selected the tail 50,000 rows (`df.tail(50000)`)[cite: 40, 64].
* [cite_start]**Image Path Creation:** Constructed local image paths by extracting the filename from the `image_link` URL[cite: 69].
* **Corrupt Image Removal:** Implemented a rigorous two-stage filtering process:
    1.  [cite_start]Verified that the image file existed on disk[cite: 71, 881].
    2.  [cite_start]Verified that the image file could be successfully decoded by TensorFlow to check for corruption[cite: 409, 770].
* [cite_start]**Resulting Clean Data:** The final dataset used for training comprised **19,852** samples[cite: 422, 436].

### 2. Feature Engineering
Numerical features were engineered from the `catalog_content` text for potential enhancements:
* [cite_start]**`pack_count`:** Extracted from the "Pack of (\d+)" pattern, defaulting to 1[cite: 368].
* [cite_start]**`item_size`:** Extracted product size in "Ounce," "oz," or "OZ"[cite: 369].
* [cite_start]**`total_size`:** Calculated as `pack_count * item_size`[cite: 371].
* [cite_start]**`unit_price`:** Calculated as `price / (total_size + \epsilon)`[cite: 372].
* [cite_start]**`brand`:** Extracted by identifying consecutive capitalized words at the start of the "Item Name"[cite: 361, 364].
* [cite_start]**Target Transformation (`log_price`):** Applied the $\log(1+x)$ transformation to the raw `price` column to normalize its distribution[cite: 340, 382].

### 3. Data Splitting
[cite_start]The clean dataset was split using a **stratified split** based on `price_bin` (10 price quantiles)[cite: 213, 385]:
* [cite_start]**Split Ratio:** $80\%$ Training, $20\%$ Validation[cite: 221, 389].
* **Final Shapes:**
    * [cite_start]Training Set: **15,881** rows [cite: 575]
    * [cite_start]Validation Set: **3,971** rows [cite: 576]

***

## 🧠 Model Architecture and Training

### Model
A **Transfer Learning** approach was used for the image feature extraction.

* [cite_start]**Base Model:** Pre-trained **EfficientNetB0** (on ImageNet)[cite: 460, 463]. [cite_start]The base model layers were **frozen**[cite: 465].
* [cite_start]**Custom Head:** The head consists of a `GlobalAveragePooling2D` layer [cite: 469][cite_start], a **Dense (128, ReLU)** layer [cite: 470][cite_start], a **Dropout (0.3)** layer [cite: 471][cite_start], and a final **Dense (1)** output layer[cite: 472].

### Training Configuration

* **Initial Training (Target: Raw Price):**
    * [cite_start]Loss: Mean Squared Error[cite: 482].
    * [cite_start]Validation $\text{MAE}$: Approximately **\$19.14** (at epoch 4)[cite: 532].
* **Final Training (Target: Log-Transformed Price - $\log\_price$):**
    * [cite_start]**Target Variable:** `log_price`[cite: 1217, 1222].
    * [cite_start]**Loss:** Mean Squared Error[cite: 1412].
    * [cite_start]**Optimizer:** Adam[cite: 481, 1411].
    * [cite_start]**Callbacks:** **Early Stopping** on `val_loss` with a `patience=3` was used[cite: 497, 1418].

***

## 💾 Submission and Artifacts

The final, logarithm-trained model was used to generate predictions on the test set.

### Prediction Process
1.  [cite_start]**Test Data Loading:** Loaded `test.csv` and created image paths using the **`sample_id`**[cite: 871, 875].
2.  **Valid Image Filtering:** Filtered the test set. [cite_start]**65,681** valid images were found out of 75,000[cite: 915].
3.  [cite_start]**Prediction:** The model predicted the **$\log\_price$** for all valid samples[cite: 897].
4.  **Inverse Transformation:** Predictions were converted to the raw price scale.
5.  [cite_start]**Submission File Creation:** Missing prices (due to invalid images) were imputed using the **mean predicted price**[cite: 905]. [cite_start]All final prices were ensured to be non-negative[cite: 907].
6.  [cite_start]The submission file `test_out.csv` contains the required **75,000 rows**[cite: 1195].

### Key Artifacts Saved
| File Name | Description | Size |
| :--- | :--- | :--- |
| `amazon_price_log_model.keras` | [cite_start]Final trained Keras model (predicts $\log(\text{price})$)[cite: 1430]. | [cite_start]~18.14 MB[cite: 1182]. |
| `train_df_final.pkl` | [cite_start]Final feature-engineered training DataFrame[cite: 1172]. | [cite_start]17.28 MB[cite: 1182]. |
| `val_df_final.pkl` | [cite_start]Final feature-engineered validation DataFrame[cite: 1174]. | [cite_start]4.39 MB[cite: 1182]. |
| `test_out.csv` | [cite_start]Final submission file (75,000 rows)[cite: 1176]. | [cite_start]1.81 MB[cite: 1182]. |

***

## 👩‍💻 How to Run

The workflow is designed for a Google Colab environment with Google Drive access for persistent storage.

### Prerequisites
1.  [cite_start]Necessary Python libraries must be installed (`tensorflow`, `pandas`, `numpy`, etc.)[cite: 1, 3, 270].
2.  [cite_start]The Amazon dataset files must be accessible (requires Google Drive mounting)[cite: 37, 39].

### Core Steps

1.  [cite_start]**Mount Google Drive** to access the data[cite: 37].
2.  [cite_start]Execute the preprocessing and data cleaning steps (filtering for valid images)[cite: 75, 420].
3.  Load the saved feature-engineered DataFrames (or re-run feature engineering).
4.  [cite_start]Re-build and compile the `model_log` to predict $\log\_price$[cite: 1406].
5.  [cite_start]Train the final model using the $\log\_price$ data pipelines[cite: 1422].
6.  [cite_start]Execute the final prediction script (Page 21) to generate the complete `test_out.csv` submission file[cite: 911].

***

*Contact: Team letsdoit, Amazon ML Challenge 2025*
