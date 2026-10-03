# Live Health Monitoring and Emergency Notification System

This project reads heart rate and SpO2 data from sensors over a serial connection, predicts the health status (Normal, Moderate or Abnormal) using machine learning, and sends emergency alerts when needed.

## Files
- `main.py`: reads serial data and predicts health status in real time
- `health_dataset_3class.csv`: dataset used to train the models

## How to run
1. Install the libraries: `pip install pandas scikit-learn matplotlib pyserial`
2. Connect the sensor and set the correct COM port in `main.py`
3. Run: `python main.py`