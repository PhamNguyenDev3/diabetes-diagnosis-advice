# Diabetes Diagnosis and Advice Application

This application uses a machine learning model to predict the likelihood of diabetes based on personal health indicators and provides health improvement advice based on the prediction results.

## Features

-   Predict the risk of diabetes based on user input.
-   Provide personalized health advice to reduce diabetes risk.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/PhamNguyenDev3/diabetes-diagnosis-advice.git
    cd diabetes-diagnosis-advice
    ```

2. **Create a virtual environment and activate it**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the model file and place it in the `models` directory**:

    - Ensure you have `random_forest_model.pkl` in the `./models/` directory.

5. **Run the application**:
    ```bash
    python app.py
    ```

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000/`.
2. Fill in the form with the required health indicators and click on the "Chẩn đoán" button to get a diagnosis.
3. Click the "Nhận lời khuyên" button to get personalized health advice based on the prediction.

## Screenshots

### Diagnosis Form

![Diagnosis Form](screenshots/image.png)

### Diagnosis Result

![Diagnosis Result](screenshots/image2.png)

## File Structure

    ```
    diabetes-diagnosis-advice/
    ├── models/
    │   └── random_forest_model.pkl
    ├── static/
    │   └── style.css
    ├── templates/
    │   └── index.html
    ├── app.py
    ├── api_client.py
    ├── requirements.txt
    └── README.md
    ```

## Explanation of the Code

### `app.py`

This is the main application file where the Flask server is set up and routes are defined.

-   **`/`**: Renders the home page with the diagnosis form.
-   **`/diagnose`**: Handles form submissions, performs the diabetes risk prediction using the machine learning model, and returns the result.
-   **`/get_advice`**: Provides health advice based on the prediction results.

### `api_client.py`

Contains the function `start_chat_and_send_message` to interact with a generative AI model for generating health advice.

### `templates/index.html`

The main HTML file for rendering the home page and diagnosis form.

### `static/style.css`

Contains the CSS styling for the web application.

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-branch-name`.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
