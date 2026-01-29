import torch
from models.bam_model import BAMModel
from config import IN_CHANNELS


def run_pipeline():
    model = BAMModel()
    model.eval()
    #dummy data
    x = torch.randn(1, IN_CHANNELS, 32, 32)

    with torch.no_grad():
        out = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    run_pipeline()
