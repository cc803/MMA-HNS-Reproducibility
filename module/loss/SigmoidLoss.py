import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Loss import Loss

class SigmoidLoss(Loss):

	def __init__(self, adv_temperature = None):
		super(SigmoidLoss, self).__init__()
		self.criterion = nn.LogSigmoid()
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False

	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score, sample_weight = None):
		pos_loss = -self.criterion(p_score).mean(dim = -1)
		if self.adv_flag:
			neg_loss = -(self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1)
		else:
			neg_loss = -self.criterion(-n_score).mean(dim = -1)
		loss = (pos_loss + neg_loss) / 2
		if sample_weight is None:
			return loss.mean()
		sample_weight = sample_weight.view(-1).to(device = loss.device, dtype = loss.dtype)
		return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)

	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()
