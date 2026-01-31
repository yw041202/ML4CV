# Person Search Project: Pedestrian Detection & Re-ID

Student ID: 1900140531 \
Student Name: Yingwen PENG \ 
Institutional Email: peng.yingwen@studio.unibo.it \

## 1. Project Overview

This project is part of the **Machine Learning for Computer Vision (ML4CV) 2025-2026** assignment. The goal is to solve the **Person Search** task by jointly addressing:

- **Pedestrian Detection**: Locating individuals within video frames.
- **Person Re-Identification (Re-ID)**: Matching detected individuals against a gallery of known identities.

## 2. System Architecture

The system follows a **two-step pipeline**:

1. **Detection Stage**: A detector (e.g., Faster R-CNN) identifies pedestrians in frames and generates bounding boxes.
2. **Re-ID Stage**: The detected regions are cropped and fed into an **Embedder** (e.g., OSNet) along with the query images.
3. **Matching**: Similarity scores are calculated between query embeddings and gallery embeddings to retrieve the most likely matches.

**Evaluation Metrics:**

- **mAP** (mean Average Precision)
- **Top-1 Accuracy**

## 3. Environment & Setup

This project is designed to run on the **Kaggle** platform.

### Dataset Configuration

To ensure the code runs correctly, please follow these steps to add the data:

1. In the Kaggle sidebar, click **"+ Add Input"**.
2. Add the dataset provided for the assignment and ensure the path is mapped to: “/kaggle/input/prw_person_re_identification_in_the_wild”.

## 4. How to Run

1. **Configure Parameters**: Locate the “ablation_configs” section within the **"Experiment Pipeline"** in main.ipynb. Modify the parameters according to your experiment requirements (Default values are the experiment settings).
2. **Hardware**: Select an **Accelerator** (GPU T4 x2) from the Kaggle settings. 
3. **Execution**: Click **"Run All"** to execute the entire pipeline.

## 5. Training Mode (Optional)

If you set `config['do_train'] = True`, additional setup is required:

- **Weights & Biases (WandB)**:
    - Go to **Add-ons -> Secrets** in the Kaggle menu.
    - Add a new secret with label “wandb_api_key” and your API key as the value.
- **File Persistence**:
    - In the right sidebar, under **"Session options"**, set **Persistence** to **"Files only"**. This ensures your trained network weights (.pth files) are saved even if the session restarts.

Note: If do_train is False, you can skip the above steps.

## 6. Results

For detailed evaluation results and visualization of the detection/retrieval, please refer to the **"Evaluation"** section in the main.ipynb notebook.
