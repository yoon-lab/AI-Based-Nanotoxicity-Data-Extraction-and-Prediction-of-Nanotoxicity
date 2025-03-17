# AI-Based-Nanotoxicity-Data-Extraction-and-Prediction-of-Nanotoxicity

This repository is based on the paper **"Automated Nanotoxicity Data Extraction and Prediction of Nanotoxicity"**. It provides the code for an AI-driven pipeline designed to share and implement the methods described in the paper for nanotoxicity data extraction and prediction.

## Overview

With the increasing use of nanomaterials, efficient nanotoxicity assessment has become essential. In this project, we implement a workflow—from data preparation to automated data extraction, prompt engineering, data preprocessing, and finally AutoML model development—by leveraging the LangChain framework and large language models (LLMs).

## Key Features

- **Automated Data Extraction**  
  - Utilizes LangChain and text embedding techniques to automatically extract nanotoxicity-related data from research articles.
  - Applies prompt engineering techniques to output structured data.
- **Data Preprocessing and Augmentation**  
  - Implements techniques such as missing value imputation and class imbalance correction (e.g., SMOTE).

## File Structure

This repository contains a total of 5 Jupyter Notebook (.ipynb) files, each tailored for a specific LLM:

- **gpt-3.5.ipynb**  
  Implements the complete workflow using the GPT-3.5 model.
- **gpt4_turbo.ipynb**  
  Implements the complete workflow using the GPT-4 Turbo model.
- **gpt-4o.ipynb**  
  Implements the complete workflow using the GPT-4o model.
- **claude_3.5_sonnet.ipynb**  
  Implements the complete workflow using the Claude 3.5 Sonnet model.
- **gemini_v2.ipynb**  
  Implements the complete workflow using the Gemini_v2 model.

Notebook covers the following steps:
1. **Data Preparation:** Load relevant research papers and datasets.
2. **Automated Data Extraction:** Use LLMs for text embedding and extraction of nanotoxicity data.
3. **Prompt Engineering:** Design detailed prompts for structured data output.
4. **Data Preprocessing:** Address missing values, handle class imbalances, and perform necessary data cleaning.

## How to Run

1. **Environment Setup**  
   - Install Python 3.7 or higher.  
   - Install the required packages (e.g., run `pip install -r requirements.txt`).
   
2. **Running the Notebooks**  
   - Open the desired Notebook file in Jupyter Notebook or Jupyter Lab and execute the cells sequentially.

3. **Reviewing the Results**  
   - Each Notebook provides visualizations and log outputs to review the results from data extraction, preprocessing, and model development.

## Reference Paper

The code in this repository is based on the following paper:

> **Automated Nanotoxicity Data Extraction and Prediction of Nanotoxicity**  
> **Authors:** Eunyong Ha, Seung Min Ha, Zayakhuu Gerelkhuu, Hyun-Yi Kim, Tae Hyun Yoon  
> For detailed information, please refer to the attached PDF.  
> :contentReference[oaicite:0]{index=0}

## Contributing

If you would like to contribute to this project, please feel free to open an issue or submit a pull request (PR) with your suggestions. Contributions in the form of code modifications, bug fixes, and feature enhancements are all welcome.

## License

This project is distributed under the [LICENSE NAME] (e.g., MIT License). For more details, please refer to the [LICENSE file](./LICENSE).

## Contact

For any questions or suggestions, please contact:  
