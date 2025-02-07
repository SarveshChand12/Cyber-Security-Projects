# Explainable Threat Hunting for SIEM Tools

This project focuses on integrating explainable AI (XAI) techniques into cybersecurity workflows, particularly within Security Information and Event Management (SIEM) tools. By leveraging machine learning models and providing interpretable explanations, the project aims to enhance threat detection and analysis capabilities for security analysts.

## Project Overview

The repository contains scripts that implement a system for detecting and explaining potential threats in network traffic data. The system processes raw network traffic data, trains a neural network model for intrusion detection, applies adversarial attacks to assess model robustness, and generates both original and adversarial packet capture (PCAP) files for analysis.

## Features

- **Data Preprocessing**: Includes feature selection, normalization, and label binarization to prepare raw network traffic data for model training.
- **Model Training**: Utilizes a neural network to classify network traffic as benign or malicious.
- **Adversarial Attacks**: Implements the Fast Gradient Sign Method (FGSM) to generate adversarial examples, assessing the model's resilience to evasion techniques.
- **Packet Crafting**: Generates original and adversarial network packets using Scapy, facilitating the analysis of how adversarial perturbations affect packet attributes.
- **Explainability**: Integrates SHAP (SHapley Additive exPlanations) and Grad-CAM (Gradient-weighted Class Activation Mapping) to provide insights into model decisions, enhancing interpretability for security analysts.

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SarveshChand12/Cyber-Security-Projects.git
   cd Cyber-Security-Projects/Explainable\ Threat\ Hunting\ for\ SIEM\ Tools
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**:

   - Place your raw network traffic data (`MachineLearningCVE.csv`) in the appropriate directory.

2. **Run the Main Script**:

   - Execute the primary script to preprocess data, train the model, generate adversarial examples, and create PCAP files:

     ```bash
     python main_script.py
     ```

3. **Analyze Results**:

   - Use tools like Wireshark to inspect the generated `original_traffic.pcap` and `adversarial_traffic.pcap` files.
   - Review the explanations provided by SHAP and Grad-CAM to understand model decisions.


- **Access**:

  Navigate to `http://127.0.0.1:8050/` in your web browser to interact with the dashboard.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the open-source community for providing tools and libraries that made this project possible.

For more information, visit the [project repository](https://github.com/SarveshChand12/Cyber-Security-Projects/tree/main/Explainable%20Threat%20Hunting%20for%20SIEM%20Tools). 