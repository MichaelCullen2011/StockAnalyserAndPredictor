class MLVars:
    """Initialises the variables needed for the ML training"""

    timescale: str
    future: int
    num_predictions: int
    epochs: int
    batch: int

    def __init__(
        self, future: int = 1, timescale: str = "days", data_type: str = "stocks"
    ):
        self.timescale: str = timescale
        self.future: int = future
        self.data_type = data_type
        self.num_predictions: int = (
            int((self.future * {"mins": 390, "days": 1}[self.timescale]))
            if data_type == "stocks"
            else int((self.future * {"mins": 60 * 24, "days": 1}[self.timescale]))
        )
        self.epochs: int = {"mins": 2, "days": 100}[self.timescale]
        self.batch: int = {"mins": 60, "days": 5}[self.timescale]


if __name__ == "__main__":
    ML = MLVars(timescale="mins", future=1)
    print(ML.data_type)
