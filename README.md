```markdown
# Traffic Density Monitoring System

This project is a **real-time traffic density monitoring system** built using **YOLOv8** for vehicle detection and **OpenCV** for video processing. The system can classify traffic density into three categories (Low, Medium, High) based on the number of vehicles detected in each frame. It also tracks different types of vehicles (cars, trucks, buses, motorbikes, bicycles) and generates HTML reports with traffic data visualizations.

## Features

- **Vehicle Detection**: Uses YOLOv8 to detect vehicles in video streams.
- **Traffic Density Classification**: Classifies traffic as Low, Medium, or High based on vehicle counts.
- **HTML Report Generation**: Summarizes traffic density and vehicle type distribution in HTML reports.
- **Legend Panel**: Displays real-time traffic metrics on the video stream.
- **Supports Video and Webcam Input**: You can either upload a video file or use your webcam for real-time monitoring.
- **Real-time Output Video**: Saves annotated video output with traffic details.

## Requirements

The system is built using Python and requires the following libraries, which can be installed via `pip`:

```bash
pip install -r requirements.txt
```

### Key Libraries:

- `opencv-python`: For video processing and display.
- `numpy`: For numerical operations.
- `ultralytics`: For the YOLOv8 model.
- `logging`: For logging events in the system.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ramyavarma123/trafficdensitymonitoring.git
cd trafficdensitymonitoring
```

### 2. Install Dependencies
Make sure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Running the System
To run the traffic monitoring system, use the following command:

```bash
python app.py
```

### 4. Choose Video Source
After running the command, you will be prompted to choose between:
1. Uploading a video file.
2. Using your webcam for real-time monitoring.

Follow the on-screen instructions to proceed.

### 5. Viewing Output
- The system will show the video stream with real-time traffic analysis.
- Press `q` to stop the monitoring.
- An annotated video and an HTML report will be generated in the specified output folder.

## Project Structure

```
trafficdensitymonitoring/
│
├── app.py        # Main Python script for monitoring traffic
├── requirements.txt          # Dependencies for the project
├── README.md                 # Project documentation
└── output/                   # Folder for generated reports and output videos
```

## Future Enhancements
- Support for additional traffic analysis metrics.
- Better tracking algorithms for vehicle re-identification.
- More advanced visualization features.

## Contributing

Feel free to fork this repository and contribute via pull requests! All contributions are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Notes:
- Replace `traffic_monitor.py` with the actual name of your main Python file if it's different.
- You can add more specific details to the "Future Enhancements" section if you have any planned features in mind.
