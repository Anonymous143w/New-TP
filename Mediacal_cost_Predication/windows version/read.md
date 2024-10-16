# Food Analysis MLOps Project

This project implements a Machine Learning Operations (MLOps) pipeline for analyzing food based on nutritional data and user queries.

## Directory Structure

```
insurance/
├── config/
│   └── config.yaml             # Configuration file for the project
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_ingestion.py   # Handles data loading and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── embeddings.py       # Generates embeddings for food descriptions
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_trainer.py    # Manages model training and saving
│   ├── utils/
│   │   ├── __init__.py
│   │   └── common.py           # Common utility functions
│   └── pipeline/
│       ├── __init__.py
│       └── training_pipeline.py # Orchestrates the entire training process
├── app/
│   └── api.py                  # Flask API for serving predictions
├── tests/
│   └── test_pipeline.py        # Unit tests for the pipeline
├── requirements.txt            # Project dependencies
├── main.py                     # Entry point for running the pipeline
└── README.md                   # This file
```

## Key Components

1. **Config**: Contains configuration files for the project.
2. **Data**: Handles data ingestion and preprocessing.
3. **Features**: Manages feature engineering, specifically generating embeddings.
4. **Models**: Handles model training and saving.
5. **Pipeline**: Orchestrates the entire training process.
6. **App**: Implements the API for serving predictions.
7. **Tests**: Contains unit tests for the pipeline.

## Setup and Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Update the `config/config.yaml` file with your specific paths and model choices.

3. Run the training pipeline:
   ```
   python main.py --config config/config.yaml
   ```

4. Start the API server:
   ```
   python -m app.api
   ```

5. Make predictions using the API:
   ```python
   import requests

   response = requests.post('http://localhost:5000/analyze', 
                           json={'query': 'Is white bread healthy?'})
   print(response.json()['response'])
   ```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
