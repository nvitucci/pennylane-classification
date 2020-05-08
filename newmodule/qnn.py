import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from pennylane import numpy as np

from newmodule import components


# Variational circuit: embedding -> variational layer -> measurement
class QNN:
    def __init__(self, dev, n_qubits, q_depth, interface="torch"):
        self.dev = dev
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.interface = interface

    def quantum_net(self):
        def quantum_net(q_input_features, q_weights_flat):
            """
            The variational quantum circuit.
            """

            # Reshape weights
            q_weights = q_weights_flat.reshape(self.q_depth, self.n_qubits)

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            components.unbiased(self.n_qubits)

            # Embed features in the quantum node
            components.embedding(q_input_features)

            # Sequence of trainable variational layers
            components.variational_layer(self.q_depth, self.n_qubits, q_weights)

            # Expectation values in the Z basis
            exp_vals = components.measurement(self.n_qubits)
            return tuple(exp_vals)

        return qml.QNode(quantum_net, self.dev, interface=self.interface)


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, qnn, device, q_delta = 0.01):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.pre_net = nn.Linear(512, qnn.n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(qnn.q_depth * qnn.n_qubits))
        self.post_net = nn.Linear(qnn.n_qubits, 2)

        self.qnn = qnn
        self.device = device

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.qnn.n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            qn = self.qnn.quantum_net()
            q_out_elem = qn(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)
