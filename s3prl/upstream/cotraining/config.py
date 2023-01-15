config = {
    "mode": "marginal",
    "num_codes": 256,
    "lstm": {
        "input_size": 40,
        "hidden_size": 512,
        "num_layers": 3,
        "batch_first": True,
        "residual": False,
    },
    "lin": {
        "in_features": 512,
        "out_features": 256,
    },
    "quantizer": {
        "input_dim": 40,
        "num_codes": 256,
        "temp": [2.0, 0.5, 0.99995],
        "code_dim": 40,
    },
    "dropout": 0.0,
}
