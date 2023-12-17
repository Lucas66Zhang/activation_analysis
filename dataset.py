from torch.utils.data import Dataset

class PairDataset_Signal(Dataset):

    def __init__(self,
                 x_maxpool2,
                 x_fc1,
                 x_fc2,
                 x_fc3,
                 ratio,
                 type,
                 ground_label,
                 pred_label):
        super(PairDataset_Signal, self).__init__()

        self.x_maxpool2 = x_maxpool2
        self.x_fc1 = x_fc1
        self.x_fc2 = x_fc2
        self.x_fc3 = x_fc3
        self.ratio = ratio
        self.type = type
        self.ground_label = ground_label
        self.pred_label = pred_label

    def __getitem__(self, index):
        return self.x_maxpool2[index], self.x_fc1[index], self.x_fc2[index], self.x_fc3[index], self.ratio[index], self.type[index], self.ground_label[index], self.pred_label[index]

    def __len__(self):
        return self.x_maxpool2.shape[0]


