# Intrusion Detection System

This project implements an Intrusion Detection System (IDS) that monitors network traffic to identify potential security threats using machine learning techniques.

## Features

- **Data Preprocessing**: Handles raw network traffic data, including feature selection, normalization, and label binarization.
- **Model Training**: Utilizes a neural network trained on preprocessed data to classify traffic as benign or malicious.
- **Adversarial Attack Simulation**: Applies the Fast Gradient Sign Method (FGSM) to generate adversarial examples, testing the robustness of the IDS.
- **Packet Crafting**: Generates original and adversarial network packets using Scapy for testing purposes.
- **Alert Comparison**: Compares the number of alerts triggered by Suricata for both original and adversarial traffic.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.8 or higher
- pandas
- numpy
- torch
- scikit-learn
- scapy
- Suricata (for alert comparison)

You can install the Python dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your network traffic data (e.g., `MachineLearningCVE.csv`) in the appropriate directory and update the `DATA_PATH` in the script accordingly.

2. **Configuration**: Adjust the configuration parameters in the script as needed, such as `SELECTED_FEATURES`, `LABEL_COLUMN`, `EPSILON` (for FGSM attack strength), and `PCAP_SIZE` (number of packets to generate).

3. **Run the Script**: Execute the main script to preprocess data, train the model, generate adversarial examples, and create PCAP files:

   ```bash
   python intrusion_detection_system.py
   ```

4. **Suricata Analysis**: Use Suricata to analyze the generated PCAP files:

   ```bash
   suricata -c /etc/suricata/suricata.yaml -r original_traffic.pcap
   suricata -c /etc/suricata/suricata.yaml -r adversarial_traffic.pcap
   ```

5. **Alert Comparison**: Review and compare the alerts generated by Suricata for both original and adversarial traffic to assess the IDS's performance.

## Notes

- The script assumes binary classification of network traffic, labeling data as either benign or malicious.
- Features are normalized before training to improve model performance.
- The FGSM attack perturbs features to test the IDS's robustness against adversarial examples.
- Scapy is used to create realistic network packets for testing purposes.
- Results should show a comparison of alert counts for original versus adversarial traffic, highlighting the impact of adversarial attacks on detection efficacy.

For more details and updates, please refer to the project's GitHub repository. 