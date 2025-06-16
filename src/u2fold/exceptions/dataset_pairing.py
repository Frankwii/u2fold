class DatasetPairingError(Exception):
    def __init__(self, errmsg: str):

        super().__init__(errmsg)
