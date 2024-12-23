import torch.nn as nn
import torch



class SumOfDistancesLoss(nn.Module):

    '''
    TO COMPLETE
    '''
    def __init__(self):
        super(SumOfDistancesLoss, self).__init__()
        
    # TO CHECK
    def forward(self, pred, target):
        distances = torch.sqrt(torch.sum((pred - target) ** 2, dim=2))

        # Sum the distances of the points in the same point cloud
        total_distances = torch.sum(distances, dim=1)

        # Mean of the total distances between batches
        mean_distance = torch.mean(total_distances)

        return mean_distance