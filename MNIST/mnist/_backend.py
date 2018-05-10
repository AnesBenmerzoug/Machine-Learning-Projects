from __future__ import print_function, division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def backendMDLSTMCell(input_, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    """
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
    """

    hx, cx, hy, cy = hidden
    gates = F.linear(input_, w_ih, b_ih) + F.linear(torch.cat((hx, hy), 1), w_hh, b_hh)

    gate_num = 5
    ingate, forgetgate1, forgetgate2, cellgate, outgate = gates.chunk(gate_num, 1)

    ingate = F.sigmoid(ingate)
    forgetgate1 = F.sigmoid(forgetgate1)
    forgetgate2 = F.sigmoid(forgetgate2)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cz = (forgetgate1 * cx) + (forgetgate2 * cy) + (ingate * cellgate)
    hz = outgate * F.tanh(cz)

    return hz, cz


def StackedMDRNN(inners, num_layers, mdlstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input_, hidden, weight):
        assert (len(weight) == total_layers)
        next_hidden = []

        if mdlstm is True:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                hy, output = inner(input_, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)
            input_ = torch.cat(all_output, input_.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input_ = F.dropout(input_, p=dropout, training=train, inplace=False)

        next_h, next_c = zip(*next_hidden)
        next_hidden = (
            torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
            torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
        )

        return next_hidden, input_

    return forward


def Recurrent(inner, direction=0):
    def forward(input_, hidden_y, weight):
        height = input_.size(0)
        width = input_.size(1)
        hidden_x = [(Variable(torch.zeros_like(hidden_y[0].data), requires_grad=False),
                     Variable(torch.zeros_like(hidden_y[1].data), requires_grad=False))]*height
        row_steps = range(height) if (direction == 0 or direction == 2) else range(height - 1, -1, -1)
        col_steps = range(width) if (direction == 0 or direction == 1) else range(width - 1, -1, -1)
        output = []

        for j in col_steps:
            if j != 0:
                hidden_y = (Variable(torch.zeros_like(hidden_y[0].data), requires_grad=False),
                            Variable(torch.zeros_like(hidden_y[1].data), requires_grad=False))
            for i in row_steps:
                hidden_y = inner(input_[i, j], (*hidden_x[i], *hidden_y), *weight)
                # Store the output and the state for the next row
                hidden_x[i] = hidden_y
                # Store the output
                output.append((i + j * height, (hidden_y[0] if isinstance(hidden_y, tuple) else hidden_y)))

        output.sort(key=lambda index_value: index_value[0])
        output = [elem[1] for elem in output]
        #output = torch.cat(output, 0).view(input_.size(1), input_.size(0), *output[0].size()).transpose(0, 1)
        output = torch.cat(output, 0).view(input_.size(0), input_.size(1), *output[0].size())

        return hidden_y, output

    return forward


def AutogradMDRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                  dropout=0, train=True, dropout_state=None, flat_weight=None):
    if mode != 'MDLSTM':
        raise NotImplementedError("Only MDLSTM has been implemented so far")
    cell = backendMDLSTMCell

    # Function that will do the recursion on fixed size inputs, no variable length inputs yet
    rec_factory = Recurrent

    # A layer of MDLSTM Cells, 4 in the 2D case
    layer = (rec_factory(cell, direction=0), rec_factory(cell, direction=1),
             rec_factory(cell, direction=2), rec_factory(cell, direction=3))

    # Function that will do all layers
    func = StackedMDRNN(layer,
                        num_layers,
                        (mode == 'MDLSTM'),
                        dropout=dropout,
                        train=train)

    def forward(input_, hidden, weight):
        if batch_first:
            input_ = input_.permute(1, 2, 0, 3)
        nexth, output = func(input_, hidden, weight)

        if batch_first:
            output = output.permute(2, 0, 1, 3).contiguous()
        return output, nexth

    return forward


def backendMDRNN(*args, **kwargs):
    def forward(input_, *fargs, **fkwargs):
        # No Cuda MDRNN yet
        func = AutogradMDRNN(*args, **kwargs)

        return func(input_, *fargs, **fkwargs)
    return forward
