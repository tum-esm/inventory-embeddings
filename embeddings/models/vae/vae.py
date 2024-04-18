class Encoder: ...


class Decoder: ...


class VariationalAutoEncoder:
    def __init__(self) -> None:
        self._encoder = Encoder()
        self._decoder = Decoder()

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def decoder(self) -> Decoder:
        return self._decoder
