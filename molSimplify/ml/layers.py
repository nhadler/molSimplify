import tensorflow as tf
from typing import List, Tuple

@tf.keras.saving.register_keras_serializable(package="molSimplify")
class PermutationLayer(tf.keras.layers.Layer):

    def __init__(self, permutations: List[Tuple[int]]):
        super().__init__()
        self.permutations = permutations

    def call(self, inputs):
        # Shape (batch_size, n_tmc_features)
        tmc_inputs = inputs["tmc"]
        # Shape (batch_size, n_ligands, n_ligand_features)
        ligand_inputs = inputs["ligand"]
        outputs = []
        for p in self.permutations:
            outputs.append(
                tf.concat(
                    [
                        tmc_inputs,
                    ]
                    + [
                        ligand_inputs[:, p_i, :] for p_i in p
                    ],
                    axis=-1,
                )
            )
        return tf.stack(outputs, axis=1)

    def get_config(self):
        return {"permutations": self.permutations}
