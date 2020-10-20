import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_dim, num_taxonomies, delta_difficulties=True):
        """
        Parameters
        ----------
        in_dim : int
            input dimension, observations dimension 
        num_taxonomies : int
            number of taxonomies, 6 in our case

        delta_difficulties : boolean
            True is we uses delta_difficulties, i.e 5 levels to represent difficulty, E, EM, M, MH, H or 3 levels E, M, H
            If true, then 5 Q-values representing the possible changes in difficulties will be
            predicted. Else, 3 Q-values representing the absolute difficulties will be predicted.

        Returns
        -------


        """
        super(Model, self).__init__()
        # Core dimensions are the inner dimensions of the neural network.
        # Adding more dimensions will add layers automatically.
        core_dims = [in_dim, 128, 128, 64, 32]
        core_arr = []
        for i in range(1, len(core_dims)):
            core_arr.extend(
                [nn.Linear(core_dims[i - 1], core_dims[i], bias=False),
                 nn.BatchNorm1d(core_dims[i]),
                 nn.ReLU()])
        self.core = nn.Sequential(*core_arr)

        # 16 VARK Actions are available which are the possible combinations for VARK: 0000, 0001, 0010, etc.
        self.vark_terminal = nn.Linear(core_dims[-1], 16)
        # We have 6 taxonomies.
        dim_taxonomy = 5 if delta_difficulties else 3
        self.num_taxonomies = num_taxonomies

        self.taxonomy_terminals = nn.ModuleList([nn.Linear(core_dims[-1], dim_taxonomy) for _ in range(self.num_taxonomies)])

    def forward(self, *input):
        """
        Parameters
        ----------
        input : numpy array
            observation  


        Returns
        -------
        vark_head : numpy array of dimension equals to VARK dimension

        taxonomy_heads : numpy array of shape equals to number of taxonomies * 5 if delta_difficulties is True, 3 otherwise

        """
        net = self.core(input[0])
        vark_head = self.vark_terminal(net)
        taxonomy_heads = []
        for i in range(self.num_taxonomies):
            taxonomy_heads.append(self.taxonomy_terminals[i](net))
        return vark_head, taxonomy_heads
