# run.py
import logging
from pipeline.training_pipeline import TrainingPipeline
import uvicorn

logging.basicConfig(level=logging.INFO)

def train():
    pipeline = TrainingPipeline("config/config.yaml")
    model = pipeline.run_pipeline()
    logging.info("Training completed successfully")

def serve():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "serve":
            serve()
    else:
        print("Please specify 'train' or 'serve'")