# Plant Disease Classifier

An AI-powered app that helps identify plant diseases using images.

## Overview

The Plant Disease Classifier is a machine learning-based application designed to help farmers and gardeners identify plant diseases through image recognition. The model is trained on a diverse dataset of plant images to accurately classify various diseases.

## Features

- **AI-powered image classification**: Utilizes advanced machine learning algorithms to identify plant diseases.
- **Multi-disease identification**: Capable of recognizing multiple plant diseases.
- **User-friendly interface**: Easy-to-use interface for uploading and analyzing images.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ayushpratapsingh1/Plant-Disease-Classifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Plant-Disease-Classifier
    ```
3. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open `PlantDiseaseClassifier.ipynb` and follow the instructions to train the model on your dataset.

### Running the Model

To run the model on new images:

1. Ensure your trained model is saved.
2. Use the following script to classify new images:
    ```python
    import your_model_module  # Replace with your actual module
    from PIL import Image

    model = your_model_module.load_model('path_to_your_model')  # Replace with your model path
    image = Image.open('path_to_image')  # Replace with your image path
    prediction = model.predict(image)
    print(f'The predicted disease is: {prediction}')
    ```

## Dataset

The dataset used for training the model should contain images of plants with and without diseases. Ensure that the images are organized into folders representing each class. You can use publicly available datasets or create your own by collecting images from various sources.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m 'Add new feature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- Inspired by various machine learning and computer vision projects.
- Thanks to all contributors and the community for their support.
