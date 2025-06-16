class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                    nn.BatchNorm2d(out_channels),
                    nn.ELU(inplace=True))
        
    def forward(self, x):
        x = self.conv(x)
        return x
        
class Unet(nn.Module):
    
    def __init__(self):
        super(Unet, self).__init__()
             
        self.double_conv1 = double_conv(1, start_fm, 3, 1, 1)
        self.maxpool1 = nn.Sequential(
                        nn.Conv2d(start_fm, start_fm, 2, 2, 0, bias=True),
                        nn.BatchNorm2d(start_fm),
                        nn.ELU(inplace=True))
        self.double_conv2 = double_conv(start_fm, start_fm * 2, 3, 1, 1)
        self.maxpool2 = nn.Sequential(
                    nn.Conv2d(start_fm*2, start_fm*2, 2, 2, 0, bias=True),
                    nn.BatchNorm2d(start_fm*2),
                    nn.ELU(inplace=True))
        self.double_conv3 = double_conv(start_fm * 2, start_fm * 4, 3, 1, 1)
        self.maxpool3 = nn.Sequential(
                    nn.Conv2d(start_fm*4, start_fm*4, 2, 2, 0, bias=True),
                    nn.BatchNorm2d(start_fm*4),
                    nn.ELU(inplace=True))
        self.double_conv4 = double_conv(start_fm * 4, start_fm * 8, 3, 1, 1)
        self.maxpool4 = nn.Sequential(
                nn.Conv2d(start_fm*8, start_fm*8, 2, 2, 0, bias=True),
                nn.BatchNorm2d(start_fm*8),
                nn.ELU(inplace=True))
        self.double_conv5 = double_conv(start_fm * 8, start_fm * 16, 3, 1, 1)
        
        self.t_conv4 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        self.ex_double_conv4 = double_conv(start_fm * 16, start_fm * 8, 3, 1, 1)
        self.t_conv3 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        self.ex_double_conv3 = double_conv(start_fm * 8, start_fm * 4, 3, 1, 1)
        self.t_conv2 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        self.ex_double_conv2 = double_conv(start_fm * 4, start_fm * 2, 3, 1, 1)
        self.t_conv1 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        self.ex_double_conv1 = double_conv(start_fm * 2, start_fm, 3, 1, 1)
        self.ex_double_conv0 = double_conv(start_fm, 16, 3, 1, 1)
        self.final = nn.Sequential(
                    nn.Conv2d(16, 16, kernel_size=1, padding=0, bias=True),
                    nn.BatchNorm2d(16),
                    nn.Sigmoid())

        
    def forward(self, inputs):
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        maxpool1 = nn.Dropout(0.20)(maxpool1)
        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        maxpool2 = nn.Dropout(0.20)(maxpool2)
        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        maxpool3 = nn.Dropout(0.20)(maxpool3)
        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        maxpool4 = nn.Dropout(0.20)(maxpool4)
        conv5 = self.double_conv5(maxpool4)
        t_conv4 = self.t_conv4(conv5)
        t_conv4 = nn.Dropout(0.20)(t_conv4)
        cat4 = torch.cat([conv4 ,t_conv4], 1)
        ex_conv4 = self.ex_double_conv4(cat4)
        t_conv3 = self.t_conv3(ex_conv4)
        t_conv3 = nn.Dropout(0.20)(t_conv3)
        cat3 = torch.cat([conv3 ,t_conv3], 1)
        ex_conv3 = self.ex_double_conv3(cat3)
        t_conv2 = self.t_conv2(ex_conv3)
        t_conv2 = nn.Dropout(0.20)(t_conv2)
        cat2 = torch.cat([conv2 ,t_conv2], 1)
        ex_conv2 = self.ex_double_conv2(cat2)
        t_conv1 = self.t_conv1(ex_conv2)
        t_conv1 = nn.Dropout(0.20)(t_conv1)
        cat1 = torch.cat([conv1, t_conv1], 1)
        ex_conv1 = self.ex_double_conv1(cat1)
        ex_conv0 = self.ex_double_conv0(ex_conv1)
        result = self.final(ex_conv0)   
        return result
