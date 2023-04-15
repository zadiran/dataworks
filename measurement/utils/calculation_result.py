class calculation_result:
    def __init__(self) -> None:
        self.name : str
        self.value : float

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value