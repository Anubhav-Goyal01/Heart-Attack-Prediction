# Heart Attack Prediction

This project aims to predict the chances of a person getting a heart attack or not based on certain features like Age, Gender, Maximum Heart rate reached, cholestrol levels, blood sugar levels, type of chest pain, etc.   The model has been trained on historical data 300 patients.


![Preview](/static/preview.png)

## File Structure

- `Data`: This folder contains the dataset used for training the model.
- `src`: This folder contains the source code for the project.
- `static`: This folder contains static files used by the web app.
- `templates`: This folder contains HTML templates used by the web app.
- `.gitignore`: This file specifies the files and folders that should be ignored by Git.
- `LICENSE`: This file contains the license information for the project.
- `README.md`: This file contains the documentation for the project.
- `app.py`: This file contains the code for the Flask web app.
- `requirements.txt`: This file contains the list of Python dependencies required to run the project.
- `pipeline.py`: This file should be run before app.py, as it creates the preprocessing object and trains the machine learning model.
- `setup.py`: This file contains information about the project, such as its name, version, and dependencies.


## Usage
1. Clone this repository to your local machine.
2. Install the required dependencies by running pip install -r requirements.txt.
3. Run pipeline.py to preprocess the data and train the machine learning model.
4. Run app.py to start the Flask web app.
5. Open a web browser and navigate to http://localhost:5000.
6. Enter the required details to get probability of getting a heart attack

> `NOTE:` Make sure to run pipeline.py before running app.py to ensure that the machine learning model has been trained and is ready for use in the web app.