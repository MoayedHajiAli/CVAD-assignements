import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from GNNs import MLP, Global_Graph, Sub_Graph
from utils import (batch_init, get_dis_point_2_points, merge_tensors,
                   to_origin_coordinate)


class VectorNet(nn.Module):
    def __init__(self, hidden_size, device):
        super(VectorNet, self).__init__()

        self.sub_graph = Sub_Graph(hidden_size)
        self.global_graph = Global_Graph(hidden_size)
        self.predict_traj = MLP(hidden_size, 6*30*2 + 6)
        self.device = device
        self.hidden_size = hidden_size
        self.traj_completion_criterion = nn.SmoothL1Loss()
        self.traj_selection_criterion = nn.NLLLoss()

    def forward_encode_sub_graph(self, mapping, matrix, polyline_spans, batch_size):
        # len(input_list_list) = batch_size
        # len(input_list_list[i]) = #polylines in scene i
        # input_list_list[i][j] = #vectors in polyline j of scene i
        input_list_list = []
        # print("BS:", batch_size)
        for i in range(batch_size):
            input_list = []
            # print("Polyline len:", len(polyline_spans[i]))
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(
                    matrix[i][polyline_span], device=self.device)
                # print("node shape", tensor.shape)
                input_list.append(tensor)

            input_list_list.append(input_list)

        element_states_batch = []

        # Aim is getting a feature for each polyline
        #   from many vectors of it
        for batch_idx in range(batch_size):
            hidden_states, lengths = merge_tensors(
                input_list_list[batch_idx], self.device, self.hidden_size)
            # input_list_list[i] -> (8, 64), (6, 64), (19, 64)
            # hidden_states -> (3, 19, 64)
            # hidden_states -> (#polylines, max(#node), hidden_dim)

            

            hidden_states = self.sub_graph(hidden_states, lengths)
            # hidden_states.shape = (#polylines, hidden_size)
            # print("hidden state", hidden_states.shape)
            element_states_batch.append(hidden_states)

        return element_states_batch

    def forward(self, mapping, validate=False):

        matrix = [i["matrix"] for i in mapping]
        polyline_spans = [i["polyline_spans"] for i in mapping]
        labels = [i["labels"] for i in mapping]

        batch_init(mapping)
        batch_size = len(matrix)

        element_states_batch = self.forward_encode_sub_graph(
            mapping, matrix, polyline_spans, batch_size)

        inputs, inputs_lengths = merge_tensors(
            element_states_batch, self.device, self.hidden_size)
        # inputs.shape = (batch_size, max(#polylines), hidden_size)

        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros(
            [batch_size, max_poly_num, max_poly_num], device=self.device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask=attention_mask, mapping=mapping)

        outputs = self.predict_traj(hidden_states[:, 0, :]) # ego vehicle is at position 0
        pred_logits = outputs[:, -6:]
        pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
        outputs = outputs[:, :- 6].view([batch_size, 6, 30, 2]) # B x size_pred x ts x offset

        ### YOUR CODE HERE ###
        loss = None
        for i in range(batch_size):
            gt_points = torch.tensor(np.array(labels[i]).reshape([30, 2])).to(outputs.device)
            # print("gt_points", gt_points.shape)
            dist = torch.abs(gt_points[-1, 0] - outputs[i, :, -1, 0]) ** 2 +  torch.abs(gt_points[-1, 1] - outputs[i, :, -1, 1]) ** 2 # 6 x 2
            # print("dist", dist.shape)
            argmin = torch.argmin(dist) # find prediction with closest endpoint to gt
            re_loss = self.traj_completion_criterion(gt_points, outputs[i, argmin, :, :])  # find loss over predicted trajectory argmin
            # print("pred prop:", pred_probs[i].shape) # 6
            ce_loss = self.traj_selection_criterion(pred_probs[i].unsqueeze(0), torch.Tensor([argmin]).long().to(pred_probs.device))

            # node loss
            loss = re_loss + 0.5 * ce_loss if loss is None else loss + re_loss + 0.5 * ce_loss 
        
        loss /= batch_size

        ### YOUR CODE HERE ###
        if validate:
            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(
            ), dtype=np.float32) if pred_probs is not None else pred_probs
            for i in range(batch_size):
                for each in outputs[i]:
                    to_origin_coordinate(each, i)

            return outputs, pred_probs

        return loss
