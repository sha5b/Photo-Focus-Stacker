# Focus Stacking Tool

A Python-based focus stacking tool that combines multiple images with different focus points into a single, fully-focused image.

## Features

- Load multiple images with different focus points
- Automatic image alignment
- Advanced focus stacking algorithm
- Support for various output formats (TIFF, PNG, JPEG)
- Color space selection (sRGB, Adobe RGB, ProPhoto RGB)
- Simple and intuitive user interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Focus-Stacking.git
cd Focus-Stacking
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python src/main.py
```

1. Click "Load Images" to select your focus stack images
2. Choose your desired output format and color space
3. Click "Process Stack" to generate the final image
4. Save the result using "Save Image"

## How it Works

The focus stacking process involves several steps:

1. Image Loading & Preprocessing
2. Image Alignment
3. Focus Measure Calculation
4. Depth Map Generation
5. Image Fusion
6. Post-processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
