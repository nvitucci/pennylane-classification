import pennylane as qml

from newmodule import layers


def unbiased(n_qubits):
    return layers.H_layer(n_qubits)


def embedding(q_input_features):
    # Could use AngleEmbedding template?
    layers.RY_layer(q_input_features)


def variational_layer(q_depth, n_qubits, q_weights):
    for k in range(q_depth):
        layers.entangling_layer(n_qubits)
        layers.RY_layer(q_weights[k])


def measurement(n_qubits):
    return [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
