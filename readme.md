# Python Object Detection

A Python-based project for detecting objects in images and videos using popular deep learning frameworks.

## Features

- Detects multiple object classes in images and videos
- Easy-to-use command-line interface
- Supports model customization and retraining
- Outputs annotated images with bounding boxes

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/)
- OpenCV
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Detect objects in an image:

```bash
python detect.py --image path/to/image.jpg
```

Detect objects in a video:

```bash
python detect.py --video path/to/video.mp4
```

## Configuration

- Edit `config.yaml` to set model parameters and detection thresholds.

## Model Training

To train a custom model, follow the instructions in [`TRAINING.md`](TRAINING.md).

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.