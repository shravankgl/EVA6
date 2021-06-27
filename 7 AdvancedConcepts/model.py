import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,padding=1),  # 28x28 output 26x26 RF : 3x3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(0.05),
           
            nn.Conv2d(32, 32, 3,padding=1), # 26x26 output 24x24 RF : 5x5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(0.05),
           
            nn.Conv2d(32, 32, 3,padding=1), # 24x24 output 22x22 RF : 7x7
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
        
        )

        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = 2, padding = 1),
            # nn.Conv2d(24, 12, 1), # 24x24 output 22x22 RF : 7x7
            nn.ReLU(),
            nn.BatchNorm2d(32),

            #nn.AvgPool2d(2, 2),  # 22x22 output - 11x11 RF 14x14

        )

        self.conv2 =  nn.Sequential(

            nn.Conv2d(32, 32, 3,padding=1), # 11x11 output - 9x9 RF 16x16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(0.05),
            
            nn.Conv2d(32, 32, 3,padding=1),  # 9x9 output - 7x7 RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            
        )

        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = 2, padding = 1),
            #nn.Conv2d(32, 16, 1), # 9x9 output - 7x7 RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.AvgPool2d(2, stride=2, padding=0)
        )

        self.conv3 =  nn.Sequential(
            #Depthwise 1
            nn.Conv2d(32, 32, 3,padding=1,groups=32), # 11x11 output - 9x9 RF 16x16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(0.05),
            
            nn.Conv2d(32, 32, 3,padding=1),  # 9x9 output - 7x7 RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            
        )

        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = 2, padding = 1),
            #nn.Conv2d(32, 16, 1), # 9x9 output - 7x7 RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.AvgPool2d(2, stride=2, padding=0)
        )

        self.conv4 =  nn.Sequential(
            #Depthwise 2
            nn.Conv2d(32, 32, 3,padding=1,groups=32), # 11x11 output - 9x9 RF 16x16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(0.05),
            
            nn.Conv2d(32, 32, 3,padding=1),  # 9x9 output - 7x7 RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            
        )

        self.trans4 = nn.Sequential(

            nn.Conv2d(32, 16, 1), # 9x9 output - 7x7 RF 18x18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(4)
        )


        #self.fc = nn.Linear(32*2*2,10)
        self.outblock = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x) + x
        x = self.trans2(x)
        x = self.conv3(x) +x
        x = self.trans3(x)
        x = self.conv4(x)
        x = self.trans4(x)
        x = self.outblock(x)
        x = x.view(-1,10)
        #x = self.fc(x)
        return x
