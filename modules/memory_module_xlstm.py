"""
Memory Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import copy
from typing import Callable, Dict, Tuple
import math
import torch
from torch import Tensor
from torch.nn import GRUCell, RNNCell, Linear, LSTMCell
import torch.nn as nn
from torch_geometric.nn.inits import zeros, ones
from torch_geometric.utils import scatter
from math import sqrt
from torch import exp, tanh, sigmoid
from einops import einsum, rearrange
from torch.nn.functional import silu, gelu
from tgb.utils.utils import CausalConv1d, BlockLinear, enlarge_as

from modules.time_enc import TimeEncoder, TimeEncoderMixer

TGNMessageStoreType = Dict[int, Tuple[Tensor, Tensor, Tensor, Tensor]]



class sLSTM(nn.Module):
    '''The scalar-Long Short Term Memory (sLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model is a variant of the standard LSTM model and
    offers two major improvements:
    - Exponential gating with appropriate state normalization
        to avoid overflows induced by the exponential function.
    - A new memory mixing within heads but not across heads.
    '''
    
    def __init__(
        self,
        inp_dim : int,
        head_dim : int,
        head_num : int,
        ker_size : int = 4,
        p_factor : float = 4/3,
    ) -> None:
        super().__init__()
        
        self.inp_dim = inp_dim
        self.head_dim = head_dim
        self.head_num = head_num
        
        self.inp_norm = nn.LayerNorm(inp_dim)
        self.hid_norm = nn.GroupNorm(head_num, head_dim * head_num)
        
        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)
        
        self.W_z = nn.Linear(inp_dim, head_num * head_dim)
        self.W_i = nn.Linear(inp_dim, head_num * head_dim)
        self.W_o = nn.Linear(inp_dim, head_num * head_dim)
        self.W_f = nn.Linear(inp_dim, head_num * head_dim)
        
        self.R_z = BlockLinear([(head_dim, head_dim)] * head_num)
        self.R_i = BlockLinear([(head_dim, head_dim)] * head_num)
        self.R_o = BlockLinear([(head_dim, head_dim)] * head_num)
        self.R_f = BlockLinear([(head_dim, head_dim)] * head_num)
        
        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        # separate outputs to account for the the gated GeLU connection.
        # See Fig. 9 in the paper.
        proj_dim = int(p_factor * head_num * head_dim)
        self.up_proj   = nn.Linear(head_num * head_dim, 2 * proj_dim)
        self.down_proj = nn.Linear(proj_dim, inp_dim)
        
    @property
    def device(self) -> str:
        '''Get the device of the model.

        Returns:
            str: The device of the model.
        '''
        return next(self.parameters()).device
        
    def init_hidden(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        '''Initialize the hidden state of the sLSTM model.

        Args:
            batch_size (int): The batch size of the input sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.
        '''
        
        n_0 = torch.ones (self.head_num * self.head_dim, device=self.device)
        c_0 = torch.zeros(self.head_num * self.head_dim, device=self.device)
        h_0 = torch.zeros(self.head_num * self.head_dim, device=self.device)
        m_0 = torch.zeros(self.head_num * self.head_dim, device=self.device)
        
        return c_0, n_0, h_0, m_0
        
    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor, Tensor, Tensor],
        use_conv : bool = False,    
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        '''Forward pass of the sLSTM model.

        Args:
            seq (Tensor): The input sequence tensor of shape (batch_size, input_dim).
            hid (Tuple[Tensor, Tensor, Tensor, Tensor]): The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]: The output tensor with the residual
                connection and the newly updated hidden state tuple.
        '''
        
        b, d = seq.shape
        
        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        c_tm1, n_tm1, h_tm1, m_tm1 = hid
        
        x_t : Tensor = self.inp_norm(seq)
        
        # Optional causal convolution block for the input
        # and forget gates. See Fig. 9 in the paper.
        if use_conv:
            # FIXME: The causal conv branch is broken.
            x_c = self.causal_conv(x_t)
            x_c = silu(x_c).squeeze()
        else:
            x_c = x_t
        
        # Project the input to the different heads for all
        # the gates.
        # NOTE: For input (i) and forget (f) inputs we use
        # the output of the causal conv. See Fig. 9 in the paper.
        i_t: Tensor = self.W_i(x_c) + self.R_i(h_tm1) 
        f_t: Tensor = self.W_f(x_c) + self.R_f(h_tm1) 
        z_t: Tensor = self.W_z(x_t) + self.R_z(h_tm1)
        o_t: Tensor = self.W_o(x_t) + self.R_o(h_tm1)
        
        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_t + m_tm1, i_t)
        
        i_t = exp(i_t - m_t)         # Eq. (16) in ref. paper | or Eq. (38) in supp. mat.
        f_t = exp(f_t - m_t + m_tm1) # Eq. (17) in ref. paper | or Eq. (39) in supp. mat.
        
        z_t = tanh(z_t)              # Eq. (11) in ref. paper
        o_t = sigmoid(o_t)           # Eq. (14) in ref. paper
        
        # Update the internal states of the model
        c_t = f_t * c_tm1 + i_t * z_t # Eq. (8) in ref. paper
        n_t = f_t * n_tm1 + i_t       # Eq. (9) in ref. paper
        h_t = o_t * (c_t / n_t)       # Eq. (10) in ref. paper
        
        # Compute the output of the LSTM block
        out = self.hid_norm(h_t)
        
        # Perform up-and-down projection of the output with
        # projection factor 4/3. See Fig. (9) in supp. mat.
        out1, out2 = self.up_proj(out).chunk(2, dim=-1)
        
        out = out1 + gelu(out2)
        out = self.down_proj(out)
        
        # Return output with the residual connection and the
        # newly updated hidden state.
        return out + seq, (c_t, n_t, h_t, m_t)

class mLSTM(nn.Module):
    '''The matrix-Long Short Term Memory (mLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model is a variant of the standard LSTM model and
    offers superior memory due to its storing values in a
    matrix instead of a scalar. It is fully parallelizable
    and updates internal memory with the covariance rule.
    '''
    
    def __init__(
        self,
        inp_dim : int,
        oup_dim : int,
        head_num : int,
        head_dim : int,
        p_factor : int = 2,
        ker_size : int = 4,
    ) -> None:
        super().__init__()
        
        self.inp_dim = inp_dim
        self.head_num = head_num
        self.head_dim = head_dim
        self.oup_dim = oup_dim

        hid_dim = head_num * head_dim
        
        self.inp_norm = nn.LayerNorm(inp_dim) 
        self.hid_norm = nn.GroupNorm(head_num, hid_dim) 
        
        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        self.up_l_proj = nn.Linear(inp_dim, int(p_factor * inp_dim)) 
        self.up_r_proj = nn.Linear(inp_dim, hid_dim) 
        self.down_proj = nn.Linear(hid_dim, inp_dim) 
        self.last_proj = nn.Linear(inp_dim, oup_dim)
        
        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)
        
        self.skip = nn.Conv1d(int(p_factor * inp_dim), hid_dim, kernel_size=1, bias=False)
        
        self.W_i = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_f = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_o = nn.Linear(int(p_factor * inp_dim), hid_dim)
        
        self.W_q = nn.Linear(int(p_factor * inp_dim), hid_dim)
        self.W_k = nn.Linear(int(p_factor * inp_dim), hid_dim)
        self.W_v = nn.Linear(int(p_factor * inp_dim), hid_dim)
        
    @property
    def device(self) -> str:
        '''Get the device of the model.

        Returns:
            str: The device of the model.
        '''
        return next(self.parameters()).device
    
    def init_hidden(self, bs : int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        '''Initialize the hidden state of the sLSTM model.

        Args:
            batch_size (int): The batch size of the input sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.
        '''
        
        c_0 = torch.zeros(bs, self.head_num, self.head_dim, self.head_dim, device=self.device)
        n_0 = torch.ones (bs, self.head_num, self.head_dim               , device=self.device)
        m_0 = torch.zeros(bs, self.head_num                              , device=self.device)
        
        return c_0, n_0, m_0
    
    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        '''_summary_

        Args:
            seq (Tensor): _description_
            hid (Tuple[Tensor, Tensor]): _description_

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: _description_
        '''
        
        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        
        c_tm1, n_tm1, m_tm1 = hid
        
        x_n : Tensor = self.inp_norm(seq) # shape: b i
        
        x_t = self.up_l_proj(x_n) # shape: b (i * p_factor)
        r_t = self.up_r_proj(x_n) # shape: b (h d)
        
        # Compute the causal convolutional input (to be 
        # used for the query and key gates)
        x_c = self.causal_conv(x_t) # shape: b 1 (i * p_factor)
        #if x_c.dim() == 3: # include to avoid error with batch size 1
        x_c = silu(x_c).squeeze(1) # shape: b (i * p_factor)
        
        q_t = rearrange(self.W_q(x_c), 'b (h d) -> b h d', h=self.head_num)
        k_t = rearrange(self.W_k(x_c), 'b (h d) -> b h d', h=self.head_num) / sqrt(self.head_dim)
        v_t = rearrange(self.W_v(x_t), 'b (h d) -> b h d', h=self.head_num)
        
        i_t: Tensor = self.W_i(x_c) # shape: b h
        f_t: Tensor = self.W_f(x_c) # shape: b h
        o_t: Tensor = self.W_o(x_t) # shape: b (h d)
        
        # Compute the gated outputs for the newly computed inputs
        
        m_t = torch.max(f_t + m_tm1, i_t)
        
        i_t = exp(i_t - m_t)         # Eq. (25) in ref. paper
        f_t = exp(f_t - m_t + m_tm1) # Eq. (26) in ref. paper
        o_t = sigmoid(o_t)           # Eq. (27) in ref. paper
        
        # Update the internal states of the model
        c_t = enlarge_as(f_t, c_tm1) * c_tm1 + enlarge_as(i_t, c_tm1) * einsum(v_t, k_t, 'b h d, b h p -> b h d p')
        n_t = enlarge_as(f_t, n_tm1) * n_tm1 + enlarge_as(i_t, k_t)   * k_t                    
        h_t = o_t * rearrange(
                einsum(c_t, q_t, 'b h d p, b h p -> b h d') /
                einsum(n_t, q_t, 'b h d, b h d -> b h').clamp(min=1).unsqueeze(-1),
                'b h d -> b (h d)'
            ) # Eq. (21) in ref. paper

        x_c = rearrange(x_c, 'b i -> b i 1')
        out = self.hid_norm(h_t) + self.skip(x_c).squeeze() # shape: b (h d)
        out = out * silu(r_t)                               # shape: b (h d)
        out = self.down_proj(out)                           # shape: h i
        
        # Return output with the residual connection and the
        # newly updated hidden state.
        return self.last_proj(out + x_n), (c_t, n_t, m_t)
    
    def reset_parameters(self):
        '''_summary_'''
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

##############################################################################################################
##############################################################################################################

class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        memory_updater_cell: str = "mlstm",
        layers: int = 2,
        head_dim: int = 4,
        head_num: int = 2,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.memory_updater_cell = memory_updater_cell # new added for sLSTM & mLSTM

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoderMixer(time_dim)
        # self.intra_bs_time_enc = TimeEncoderMixer(time_dim)
        
        # self.gru = GRUCell(message_module.out_channels, memory_dim)
        if memory_updater_cell == "gru":  # for TGN
            self.memory_updater = GRUCell(message_module.out_channels, memory_dim) # (320, 100)
        elif memory_updater_cell == "rnn":  # for JODIE & DyRep
            self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
        elif memory_updater_cell == "slstm":
            self.memory_updater = sLSTM(message_module.out_channels, memory_dim, layers)
        elif memory_updater_cell == "mlstm":
            self.memory_updater = torch.compile(mLSTM(message_module.out_channels, memory_dim, head_num, head_dim))
        else:
            raise ValueError(
                "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
            )
        
        self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
        
        if memory_updater_cell == "slstm":
            self.register_buffer("hidden_state", torch.empty(num_nodes, layers, 2, memory_dim)) # 2 for h and c
        elif memory_updater_cell == "mlstm":
            self.register_buffer("hidden_state", torch.empty(num_nodes, head_num, head_dim, head_dim))
            self.register_buffer("n_state", torch.empty(num_nodes, head_num, head_dim))
            self.register_buffer("m_state", torch.empty(num_nodes, head_num))

        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer("last_update", last_update)
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()


    @property
    def device(self) -> torch.device:
        return self.time_enc.lin.weight.device

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.msg_s_module, "reset_parameters"):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, "reset_parameters"):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.memory_updater.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.hidden_state)
        zeros(self.last_update)
        if self.memory_updater_cell == "mlstm":
            ones(self.n_state)
            zeros(self.m_state)
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()
        if self.memory_updater_cell == "slstm" or self.memory_updater_cell == "mlstm":
            self.hidden_state.detach_()
        if self.memory_updater_cell == "mlstm":
            self.n_state.detach_()
            self.m_state.detach_()

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update, *arg = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]
        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg)`."""
        n_id = torch.cat([src, dst]).unique()

        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def _reset_message_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_memory(self, n_id: Tensor):
        memory, last_update, *arg = self._get_updated_memory(n_id)
        
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update
        if self.memory_updater_cell == "slstm" or self.memory_updater_cell == "mlstm":
            self.hidden_state[n_id] = arg[0]
        if self.memory_updater_cell == "mlstm":
            self.hidden_state[n_id] = arg[0]
            self.n_state[n_id] = arg[1]
            self.m_state[n_id] = arg[2]

    def _get_updated_memory(self, n_id: Tensor,) -> Tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self._compute_msg(
            n_id, self.msg_s_store, self.msg_s_module,
        )

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self._compute_msg(
            n_id, self.msg_d_store, self.msg_d_module,
        )
        
        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        
        # TODO get intra-batch relative time encoding
        # rel_bs_t = t - self.last_update[idx]
        
        # rel_bs_t_enc = self.intra_bs_time_enc(rel_bs_t.to(msg.dtype))
        
        # message aggregation by retriving the last message for each node
        # Get local copy of updated memory.
        aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

        # Get local copy of updated `last_update`.
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id] #TODO check if this is correct according to SetTransformerAggregation

        if self.memory_updater_cell == "gru" or self.memory_updater_cell == "rnn":
            memory = self.memory_updater(aggr, self.memory[n_id])
            return memory, last_update
        elif self.memory_updater_cell == "slstm":
            aggr = aggr.unsqueeze(0)
            memory, hidden_state = self.memory_updater(aggr, self.hidden_state[n_id]) # mem_slstm(input, (h, c))
            return memory.squeeze(1), last_update, hidden_state # memory dim [unique_nodes, memory_dim], ex. [34, 100]
        elif self.memory_updater_cell == "mlstm":
            memory, (hidden_state, n_state, m_state) = self.memory_updater(aggr, (self.hidden_state[n_id], self.n_state[n_id], self.m_state[n_id]))
            return memory, last_update, hidden_state, n_state, m_state

    def _update_msg_store(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        raw_msg: Tensor,
        msg_store: TGNMessageStoreType,
    ):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(
        self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable,
    ): # msg_module is IdentityMessage by concat
        # TODO msg_store is empty {}. Need to check if it is updated
        if len(n_id) > 0:
            data = [msg_store[i] for i in n_id.tolist()] # from the message store, get the messages for each uniqe node
        else:
            data = []
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc) # TODO get the right message from the message module
        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        super().train(mode)

# class sLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
#         super(sLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout

#         self.lstms = nn.ModuleList([nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
#         self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

#         self.exp_forget_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
#         self.exp_input_gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         for lstm in self.lstms:
#             nn.init.xavier_uniform_(lstm.weight_ih)
#             nn.init.xavier_uniform_(lstm.weight_hh)
#             nn.init.zeros_(lstm.bias_ih)
#             nn.init.zeros_(lstm.bias_hh)
        
#         for gate in self.exp_forget_gates + self.exp_input_gates:
#             nn.init.xavier_uniform_(gate.weight)
#             nn.init.zeros_(gate.bias)

#     def forward(self, input_seq, hidden_state=None):
#         batch_size = input_seq.size(0)
#         seq_length = input_seq.size(1)

#         if hidden_state is None:
#             hidden_state = self.init_hidden(batch_size)

#         # iterate through sequence
#         output_seq = []
#         for t in range(batch_size):
#             x = input_seq[t, :, :]
#             new_hidden_state = []
            
#             # iterate through layers
#             for l, (lstm, dropout, f_gate, i_gate) in enumerate(zip(self.lstms, self.dropout_layers, self.exp_forget_gates, self.exp_input_gates)):
#                 if hidden_state[l][0] is None:
#                     h, c = lstm(x)
#                 else:
#                     h, c = lstm(x, (hidden_state[:,l,0,:], hidden_state[:,l,1,:]))

#                 f = torch.tanh(f_gate(h))
#                 i = torch.tanh(i_gate(h))
#                 c = f * c + i * lstm.weight_hh.new_zeros(batch_size, self.hidden_size)
                
#                 new_hidden_state.append(torch.cat([h, c], dim=-1))

#                 if l < self.num_layers:
#                     x = dropout(h)
#                 else:
#                     x = h
#             hidden_state = new_hidden_state # dim = [batch_size, num_layers, 2, hidden_size]
#             output_seq.append(x)

#         output_seq = torch.stack(output_seq, dim=1)
        
#         return output_seq, torch.cat(hidden_state).view(-1, self.num_layers, 2, self.hidden_size)

#     def init_hidden(self, batch_size):
#         hidden_state = []
#         for lstm in self.lstms:
#             h = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
#             c = torch.zeros(batch_size, self.hidden_size, device=lstm.weight_ih.device)
#             hidden_state.append([h, c])
#         return hidden_state

# class TGNMemoryMixer(torch.nn.Module):
#     r"""The Temporal Graph Network (TGN) memory model from the
#     `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
#     <https://arxiv.org/abs/2006.10637>`_ paper.

#     .. note::

#         For an example of using TGN, see `examples/tgn.py
#         <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
#         tgn.py>`_.

#     Args:
#         num_nodes (int): The number of nodes to save memories for.
#         raw_msg_dim (int): The raw message dimensionality.
#         memory_dim (int): The hidden memory dimensionality.
#         time_dim (int): The time encoding dimensionality.
#         message_module (torch.nn.Module): The message function which
#             combines source and destination node memory embeddings, the raw
#             message and the time encoding.
#         aggregator_module (torch.nn.Module): The message aggregator function
#             which aggregates messages to the same destination into a single
#             representation.
#     """

#     def __init__(
#         self,
#         num_nodes: int,
#         raw_msg_dim: int,
#         memory_dim: int,
#         time_dim: int,
#         message_module: Callable,
#         aggregator_module: Callable,
#         memory_updater_cell: str = "gru",
#     ):
#         super().__init__()

#         self.num_nodes = num_nodes
#         self.raw_msg_dim = raw_msg_dim
#         self.memory_dim = memory_dim
#         self.time_dim = time_dim

#         self.msg_s_module = message_module
#         self.msg_d_module = copy.deepcopy(message_module)
#         self.aggr_module = aggregator_module
#         self.time_enc = TimeEncoderMixer(time_dim)
#         # self.gru = GRUCell(message_module.out_channels, memory_dim)
#         if memory_updater_cell == "gru":  # for TGN
#             self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
#         elif memory_updater_cell == "rnn":  # for JODIE & DyRep
#             self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
#         else:
#             raise ValueError(
#                 "Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'."
#             )

#         self.register_buffer("memory", torch.empty(num_nodes, memory_dim))
#         last_update = torch.empty(self.num_nodes, dtype=torch.long)
#         self.register_buffer("last_update", last_update)
#         self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

#         self.msg_s_store = {}
#         self.msg_d_store = {}

#         self.reset_parameters()

#     @property
#     def device(self) -> torch.device:
#         return self.time_enc.lin.weight.device

#     def reset_parameters(self):
#         r"""Resets all learnable parameters of the module."""
#         if hasattr(self.msg_s_module, "reset_parameters"):
#             self.msg_s_module.reset_parameters()
#         if hasattr(self.msg_d_module, "reset_parameters"):
#             self.msg_d_module.reset_parameters()
#         if hasattr(self.aggr_module, "reset_parameters"):
#             self.aggr_module.reset_parameters()
#         self.time_enc.reset_parameters()
#         self.memory_updater.reset_parameters()
#         self.reset_state()

#     def reset_state(self):
#         """Resets the memory to its initial state."""
#         zeros(self.memory)
#         zeros(self.last_update)
#         self._reset_message_store()

#     def detach(self):
#         """Detaches the memory from gradient computation."""
#         self.memory.detach_()

#     def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
#         """Returns, for all nodes :obj:`n_id`, their current memory and their
#         last updated timestamp."""
#         if self.training:
#             memory, last_update = self._get_updated_memory(n_id)
#         else:
#             memory, last_update = self.memory[n_id], self.last_update[n_id]

#         return memory, last_update

#     def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor):
#         """Updates the memory with newly encountered interactions
#         :obj:`(src, dst, t, raw_msg)`."""
#         n_id = torch.cat([src, dst]).unique()

#         if self.training:
#             self._update_memory(n_id)
#             self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
#             self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
#         else:
#             self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
#             self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
#             self._update_memory(n_id)

#     def _reset_message_store(self):
#         i = self.memory.new_empty((0,), device=self.device, dtype=torch.long)
#         msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
#         # Message store format: (src, dst, t, msg)
#         self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
#         self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

#     def _update_memory(self, n_id: Tensor):
#         memory, last_update = self._get_updated_memory(n_id)
#         self.memory[n_id] = memory
#         self.last_update[n_id] = last_update

#     def _get_updated_memory(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
#         self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

#         # Compute messages (src -> dst).
#         msg_s, t_s, src_s, dst_s = self._compute_msg(
#             n_id, self.msg_s_store, self.msg_s_module
#         )

#         # Compute messages (dst -> src).
#         msg_d, t_d, src_d, dst_d = self._compute_msg(
#             n_id, self.msg_d_store, self.msg_d_module
#         )

#         # Aggregate messages.
#         idx = torch.cat([src_s, src_d], dim=0)
#         msg = torch.cat([msg_s, msg_d], dim=0)
#         t = torch.cat([t_s, t_d], dim=0)
#         aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

#         # Get local copy of updated memory.
#         memory = self.memory_updater(aggr, self.memory[n_id])

#         # Get local copy of updated `last_update`.
#         dim_size = self.last_update.size(0)
#         last_update = scatter(t, idx, 0, dim_size, reduce="max")[n_id]

#         return memory, last_update

#     def _update_msg_store(
#         self,
#         src: Tensor,
#         dst: Tensor,
#         t: Tensor,
#         raw_msg: Tensor,
#         msg_store: TGNMessageStoreType,
#     ):
#         n_id, perm = src.sort()
#         n_id, count = n_id.unique_consecutive(return_counts=True)
#         for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
#             msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

#     def _compute_msg(
#         self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable
#     ):
#         data = [msg_store[i] for i in n_id.tolist()]
#         src, dst, t, raw_msg = list(zip(*data))
#         src = torch.cat(src, dim=0)
#         dst = torch.cat(dst, dim=0)
#         t = torch.cat(t, dim=0)
#         raw_msg = torch.cat(raw_msg, dim=0)
#         t_rel = t - self.last_update[src]
#         t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

#         msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

#         return msg, t, src, dst

#     def train(self, mode: bool = True):
#         """Sets the module in training mode."""
#         if self.training and not mode:
#             # Flush message store to memory in case we just entered eval mode.
#             self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
#             self._reset_message_store()
#         super().train(mode)


# class DyRepMemory(torch.nn.Module):
#     r"""
#     Based on intuitions from TGN Memory...
#     Differences with the original TGN Memory:
#         - can use source or destination embeddings in message generation
#         - can use a RNN or GRU module as the memory updater

#     Args:
#         num_nodes (int): The number of nodes to save memories for.
#         raw_msg_dim (int): The raw message dimensionality.
#         memory_dim (int): The hidden memory dimensionality.
#         time_dim (int): The time encoding dimensionality.
#         message_module (torch.nn.Module): The message function which
#             combines source and destination node memory embeddings, the raw
#             message and the time encoding.
#         aggregator_module (torch.nn.Module): The message aggregator function
#             which aggregates messages to the same destination into a single
#             representation.
#         memory_updater_type (str): specifies whether the memory updater is GRU or RNN
#         use_src_emb_in_msg (bool): whether to use the source embeddings 
#             in generation of messages
#         use_dst_emb_in_msg (bool): whether to use the destination embeddings 
#             in generation of messages
#     """
#     def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
#                  time_dim: int, message_module: Callable,
#                  aggregator_module: Callable, memory_updater_type: str,
#                  use_src_emb_in_msg: bool = False, use_dst_emb_in_msg: bool = False):
#         super().__init__()

#         self.num_nodes = num_nodes
#         self.raw_msg_dim = raw_msg_dim
#         self.memory_dim = memory_dim
#         self.time_dim = time_dim

#         self.msg_s_module = message_module
#         self.msg_d_module = copy.deepcopy(message_module)
#         self.aggr_module = aggregator_module
#         self.time_enc = TimeEncoder(time_dim)

#         assert memory_updater_type in ['gru', 'rnn'], "Memor updater can be either `rnn` or `gru`."
#         if memory_updater_type == 'gru':  # for TGN
#             self.memory_updater = GRUCell(message_module.out_channels, memory_dim)
#         elif memory_updater_type == 'rnn':  # for JODIE & DyRep
#             self.memory_updater = RNNCell(message_module.out_channels, memory_dim)
#         else:
#             raise ValueError("Undefined memory updater!!! Memory updater can be either 'gru' or 'rnn'.")
        
#         self.use_src_emb_in_msg = use_src_emb_in_msg
#         self.use_dst_emb_in_msg = use_dst_emb_in_msg

#         self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
#         last_update = torch.empty(self.num_nodes, dtype=torch.long)
#         self.register_buffer('last_update', last_update)
#         self.register_buffer('_assoc', torch.empty(num_nodes,
#                                                    dtype=torch.long))

#         self.msg_s_store = {}
#         self.msg_d_store = {}

#         self.reset_parameters()

#     @property
#     def device(self) -> torch.device:
#         return self.time_enc.lin.weight.device

#     def reset_parameters(self):
#         r"""Resets all learnable parameters of the module."""
#         if hasattr(self.msg_s_module, 'reset_parameters'):
#             self.msg_s_module.reset_parameters()
#         if hasattr(self.msg_d_module, 'reset_parameters'):
#             self.msg_d_module.reset_parameters()
#         if hasattr(self.aggr_module, 'reset_parameters'):
#             self.aggr_module.reset_parameters()
#         self.time_enc.reset_parameters()
#         self.memory_updater.reset_parameters()
#         self.reset_state()

#     def reset_state(self):
#         """Resets the memory to its initial state."""
#         zeros(self.memory)
#         zeros(self.last_update)
#         self._reset_message_store()

#     def detach(self):
#         """Detaches the memory from gradient computation."""
#         self.memory.detach_()

#     def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
#         """Returns, for all nodes :obj:`n_id`, their current memory and their
#         last updated timestamp."""
#         if self.training:
#             memory, last_update = self._get_updated_memory(n_id)
#         else:
#             memory, last_update = self.memory[n_id], self.last_update[n_id]

#         return memory, last_update

#     def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor, 
#                      embeddings: Tensor = None, assoc: Tensor = None):
#         """Updates the memory with newly encountered interactions
#         :obj:`(src, dst, t, raw_msg)`."""
#         n_id = torch.cat([src, dst]).unique()
        
#         if self.training:
#             self._update_memory(n_id, embeddings, assoc)
#             self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
#             self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
#         else:
#             self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
#             self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
#             self._update_memory(n_id, embeddings, assoc)

#     def _reset_message_store(self):
#         i = self.memory.new_empty((0, ), device=self.device, dtype=torch.long)
#         msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
#         # Message store format: (src, dst, t, msg)
#         self.msg_s_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}
#         self.msg_d_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

#     def _update_memory(self, n_id: Tensor, embeddings: Tensor = None, assoc: Tensor = None):
#         memory, last_update = self._get_updated_memory(n_id, embeddings, assoc)
#         self.memory[n_id] = memory
#         self.last_update[n_id] = last_update

#     def _get_updated_memory(self, n_id: Tensor, embeddings: Tensor = None, assoc: Tensor = None) -> Tuple[Tensor, Tensor]:
#         self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

#         # Compute messages (src -> dst).
#         msg_s, t_s, src_s, dst_s = self._compute_msg(n_id, self.msg_s_store,
#                                                      self.msg_s_module, embeddings, assoc)                                          

#         # Compute messages (dst -> src).
#         msg_d, t_d, src_d, dst_d = self._compute_msg(n_id, self.msg_d_store,
#                                                      self.msg_d_module, embeddings, assoc)

#         # Aggregate messages.
#         idx = torch.cat([src_s, src_d], dim=0)
#         msg = torch.cat([msg_s, msg_d], dim=0)
#         t = torch.cat([t_s, t_d], dim=0)
#         aggr = self.aggr_module(msg, self._assoc[idx], t, n_id.size(0))

#         # Get local copy of updated memory.
#         memory = self.memory_updater(aggr, self.memory[n_id])

#         # Get local copy of updated `last_update`.
#         dim_size = self.last_update.size(0)
#         last_update = scatter(t, idx, 0, dim_size, reduce='max')[n_id]
        
#         return memory, last_update

#     def _update_msg_store(self, src: Tensor, dst: Tensor, t: Tensor,
#                           raw_msg: Tensor, msg_store: TGNMessageStoreType):
#         n_id, perm = src.sort()
#         n_id, count = n_id.unique_consecutive(return_counts=True)
#         for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
#             msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

#     def _compute_msg(self, n_id: Tensor, msg_store: TGNMessageStoreType, msg_module: Callable, 
#                      embeddings: Tensor = None, assoc: Tensor = None):
#         data = [msg_store[i] for i in n_id.tolist()]
#         src, dst, t, raw_msg = list(zip(*data))
#         src = torch.cat(src, dim=0)
#         dst = torch.cat(dst, dim=0)
#         t = torch.cat(t, dim=0)
#         raw_msg = torch.cat(raw_msg, dim=0)
#         t_rel = t - self.last_update[src]
#         t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

#         # source nodes: retrieve embeddings
#         source_memory = self.memory[src]
#         if self.use_src_emb_in_msg and embeddings != None:
#             if src.size(0) > 0:
#                 curr_src, curr_src_idx = [], []
#                 for s_idx, s in enumerate(src):
#                     if s in n_id:
#                         curr_src.append(s.item())
#                         curr_src_idx.append(s_idx)

#                 source_memory[curr_src_idx] = embeddings[assoc[curr_src]]

#         # destination nodes: retrieve embeddings
#         destination_memory = self.memory[dst]
#         if self.use_dst_emb_in_msg and embeddings != None:
#             if dst.size(0) > 0:
#                 curr_dst, curr_dst_idx = [], []
#                 for d_idx, d in enumerate(dst):
#                     if d in n_id:
#                         curr_dst.append(d.item())
#                         curr_dst_idx.append(d_idx)
#                 destination_memory[curr_dst_idx] = embeddings[assoc[curr_dst]]
            
#         msg = msg_module(source_memory, destination_memory, raw_msg, t_enc)

#         return msg, t, src, dst

#     def train(self, mode: bool = True):
#         """Sets the module in training mode."""
#         if self.training and not mode:
#             # Flush message store to memory in case we just entered eval mode.
#             self._update_memory(
#                 torch.arange(self.num_nodes, device=self.memory.device))
#             self._reset_message_store()
#         super().train(mode)