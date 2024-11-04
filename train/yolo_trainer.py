import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings

from utils.utils import load_yaml_config, setup_logging

settings.update({"mlflow": False})


class YOLOTrainer:
    """
    A class to encapsulate YOLO model training logic.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
    ) -> None:
        """
        Initializes the YOLOTrainer class by loading a pre-trained model
        and a configuration file.

        :param config_path: Path to the training configuration YAML file.
        """
        self.config = load_yaml_config(config_path)
        self.logger = setup_logging()
        self.model = YOLO(self.config["model"]["model_path"])
        mlflow.set_tracking_uri(self.config["training"]["tracking_uri"])
        mlflow.set_experiment(self.config["training"]["experiment_name"])

    def train(self) -> None:
        """
        Trains the YOLO model based on the configuration settings.
        """
        epochs = self.config["training"]["epochs"]
        img_size = self.config["training"]["img_size"]
        batch_size = self.config["training"]["batch_size"]
        self.device = self.config["training"]["device"]

        self.logger.info(
            f"Starting training for {epochs} epochs with image size {img_size}."
        )

        # Start an MLflow run
        with mlflow.start_run(run_name="training"):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("img_size", img_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("device", self.device)

            mlflow.set_tag("Yolo model", "Testing")

            # Train the model
            results = self.model.train(
                data=self.config["data_path"],
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=self.device,
            )

            # Log training metrics (like training and validation loss)
            # mlflow.log_metric("train_loss", results.keys["box_loss"])
            # mlflow.log_metric("val_loss", results.metrics["val/box_loss"])
            for epoch in range(epochs):
                mlflow.log_metric("mAP50", results.box.map50, step=epoch)

            # Log custom metrics
            # mlflow.log_metric("mAP", mean_average_precision)

            self.logger.info("Training completed.")

            # Log the YOLO model to MLflow
            # mlflow.pytorch.log_model(
            #     self.model, artifact_path=self.config["save_model_path"]
            # )

        return results

    def evaluate(self) -> None:
        """
        Evaluates the trained model on the validation set.
        """
        self.logger.info("Starting model evaluation.")
        epochs = self.config["training"]["epochs"]

        # Start an MLflow run for evaluation
        with mlflow.start_run(run_name="validation"):
            val_results = self.model.val(device=self.device)

            # Log evaluation metrics (like accuracy or precision)
            # mlflow.log_metric("val_accuracy", val_results.metrics["precision"])
            # mlflow.log_metric("val_map", val_results.metrics["mAP50"])

            for epoch in range(epochs):
                mlflow.log_metric("mAP50", val_results.box.map50, step=epoch)

            self.logger.info("Evaluation completed.")

        return val_results

    def save_model(self) -> None:
        """
        Saves the trained model to a specified path.
        """
        self.model.save(self.config["save_model_path"])
        self.logger.info(f"Model saved at {self.config['save_model_path']}.")


if __name__ == "__main__":
    trainer = YOLOTrainer()
    trainer.train()
