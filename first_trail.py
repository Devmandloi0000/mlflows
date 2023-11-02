import mlflow
import warnings
warnings.filterwarnings('ignore')

def calculate_sum(x,y):
    return x*y


if __name__ == "__main__":
    with mlflow.start_run():
        x,y=1150,20
        z=calculate_sum(x,y)
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)