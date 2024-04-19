import numpy as np
import tensorflow as tf
from molSimplify.ml.layers import PermutationLayer


def test_permutation_layer():
    permutations = [
        (0, 1),
        (1, 0),
    ]

    layer = PermutationLayer(permutations=permutations)

    rng = np.random.default_rng(0)
    X_tmc = rng.normal(1.2, 0.4, size=(10, 4))
    X_ligand = rng.normal(1.1, 0.5, size=(10, 2, 4))

    output = layer({"tmc": X_tmc, "ligand": X_ligand})
    np.testing.assert_allclose(output, np.stack(
        [
            np.concatenate([X_tmc, X_ligand[:, 0], X_ligand[:, 1]], axis=-1),
            np.concatenate([X_tmc, X_ligand[:, 1], X_ligand[:, 0]], axis=-1),
        ], axis=1)
    )


def test_permutational_nn():
    permutations = [
        (0, 1),
        (1, 0),
    ]

    model = tf.keras.Sequential([
        PermutationLayer(permutations=permutations),
        tf.keras.layers.Dense(1),
        # Sum over the permuations
        tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1)),
    ])

    rng = np.random.default_rng(0)
    X_tmc = rng.normal(1.2, 0.4, size=(10, 4))
    X_ligand = rng.normal(1.1, 0.5, size=(10, 2, 4))

    input = {"tmc": X_tmc, "ligand": X_ligand}
    input_permuted = {"tmc": X_tmc, "ligand": X_ligand[:, [1, 0]]}
    assert model(input).shape == (10, 1)
    np.testing.assert_allclose(model(input), model(input_permuted))
