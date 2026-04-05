### For mini - project 1
import torch
from math import ceil
from time import time
from torch import nn
import matplotlib.pyplot as plt
import os
split = '\\' if '\\' in os.getcwd() else '/'

#CLIP makes the input between 0-1################################
class Clip(torch.nn.Module):
    __constants__ = ['c_min', 'c_max']
    c_min: float
    c_max: float

    def __init__(self, min: float = 0, max: float = 1):
        super().__init__()
        self.c_min = min
        self.c_max = max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clip(input, self.c_min, self.c_max)

    def extra_repr(self) -> str:
        return f'min = {self.c_min},max = {self.c_max}'
#################################################################

#Put declared modules in "parallele"#############################
class ParallelModule(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.list = nn.ModuleList(modules)

    def forward(self, input) -> torch.Tensor:
        return torch.cat([conv(input) for conv in self.list], 1)
#################################################################

#encoder-decoder class########################################### 
class ConvPoolConvTranspose2d(torch.nn.Module) :
    def __init__(self,
        in_channels, mid_channels = 32,
        kernel_sizes = [3, 5], decoder_kernel_size = 3,
        additional_encoder = 0, additional_decoder = 0,
        scale_factor = 2,
        sigma = nn.LeakyReLU(inplace = True),
        last_sigma = None,
        insides = None,
        add_innput = False,
        add_fft = 0):
        super().__init__()

        self.in_channels = in_channels

        nb_encoder = additional_encoder + 1
        nb_decoder = additional_decoder + 1

        self.kernel_sizes = kernel_sizes
        self.add_innput = add_innput
        self.add_fft = add_fft

        self.insides = insides
        self.sigma = sigma
        self.maxpull = nn.MaxPool2d(kernel_size = scale_factor)

        
        if(last_sigma != None):
            self.last_sigma = last_sigma
        else:
            self.last_sigma = self.sigma

        if(insides != None):
            assert(type(insides) == type(self))
            mid_channels = insides.in_channels
        else:
            mid_channels = in_channels

        encoder = [
            ParallelModule([
            nn.Conv2d(
            in_channels = (in_channels*3 if(add_fft > 0) else in_channels) if (layer == 0) else mid_channels,
            out_channels = mid_channels // len(kernel_sizes),
            kernel_size = kernel_size,
            padding = kernel_size // 2,
            padding_mode = 'replicate')
            for kernel_size in kernel_sizes])
            for layer in range(nb_encoder)]
        self.encoder = nn.Sequential(*[x for e in encoder for x in [e, sigma]])

        self.upsample = nn.Upsample(scale_factor = scale_factor, mode = 'bilinear', align_corners=True)

        decoder = [
            nn.Conv2d(
            in_channels = (mid_channels + (mid_channels if(self.insides != None) else 0) + (in_channels if(add_innput) else 0) + (in_channels*2 if(add_fft > 0) else 0)) if(layer == 0) else mid_channels,
            out_channels = in_channels,
            kernel_size = decoder_kernel_size,
            padding = decoder_kernel_size // 2,
            padding_mode = 'replicate')
            if (layer == nb_decoder - 1) else
            ParallelModule([
            nn.Conv2d(
            in_channels = (mid_channels + (mid_channels if(self.insides != None) else 0) + (in_channels if(add_innput) else 0) + (in_channels*2 if(add_fft > 0) else 0)) if(layer == 0) else mid_channels,
            out_channels = mid_channels // len(kernel_sizes),
            kernel_size = kernel_size,
            padding = kernel_size // 2,
            padding_mode = 'replicate')
            for kernel_size in kernel_sizes])
            for layer in range(nb_decoder)]
        decoder = [x for e in decoder for x in [e, sigma]]
        decoder[-1] = self.last_sigma
        self.decoder = nn.Sequential(*decoder)

        #Making fast fourier transform of image##################
        def fft_(input) -> torch.Tensor:
            fft = torch.fft.fft2(input)
            fft2 = torch.clone(fft)
            removed = add_fft
            fft[:,:,-removed:,:] = 0
            fft[:,:,:,-removed:] = 0
            fft[:,:,:removed+1,:] = 0
            fft[:,:,:,:removed+1] = 0
            fft2[:,:,removed+1:-removed,removed+1:-removed] = 0
            return torch.cat((torch.fft.ifft2(fft).real, torch.fft.ifft2(fft2).real),1)
        #########################################################
        if(insides != None):
            if(add_innput):
                if(add_fft > 0):
                    def forward_insides_innput_fft(input) -> torch.Tensor:
                        fft = fft_(input)
                        combined = torch.cat((input, fft),1)
                        mid = self.encoder(combined)
                        return self.decoder(torch.cat((
                            mid,
                            self.upsample(self.insides(self.maxpull(mid))),
                            combined),1))
                    self.mod = forward_insides_innput_fft
                else:
                    def forward_insides_innput(input) -> torch.Tensor:
                        mid = self.encoder(input)
                        return self.decoder(torch.cat((
                            mid,
                            self.upsample(self.insides(self.maxpull(mid))),
                            input),1))
                    self.mod = forward_insides_innput
            else:
                if(add_fft > 0):
                    def forward_insides_fft(input) -> torch.Tensor:
                        fft = fft_(input)
                        combined = torch.cat((input, fft),1)
                        mid = self.encoder(combined)
                        return self.decoder(torch.cat((
                            mid,
                            self.upsample(self.insides(self.maxpull(mid))),
                            fft),1))
                    self.mod = forward_insides_fft
                else:
                    def forward_insides(input) -> torch.Tensor:
                        mid = self.encoder(input)
                        return self.decoder(torch.cat((
                            mid,
                            self.upsample(self.insides(self.maxpull(mid)))),1))
                    self.mod = forward_insides
        else:
            if(add_innput):
                if(add_fft > 0):
                    def forward_innput_fft(input) -> torch.Tensor:
                        fft = fft_(input)
                        combined = torch.cat((input, fft),1)
                        mid = self.encoder(combined)
                        return self.decoder(torch.cat((
                            mid,
                            combined),1))
                    self.mod = forward_innput_fft
                else:
                    def forward_innput(input) -> torch.Tensor:
                        mid = self.encoder(input)
                        return self.decoder(torch.cat((
                            mid,
                            input),1))
                    self.mod = forward_innput
            else:
                if(add_fft > 0):
                    def forward_fft(input) -> torch.Tensor:
                        fft = fft_(input)
                        combined = torch.cat((input, fft),1)
                        mid = self.encoder(combined)
                        return self.decoder(torch.cat((
                            mid,
                            fft),1))
                    self.mod = forward_fft
                else:
                    def forward_(input) -> torch.Tensor:
                        return self.decoder(self.encoder(input))
                    self.mod = forward_

    def forward(self, input) -> torch.Tensor:
        return self.mod(input)
#Declaring the model#############################################
class Model(torch.nn.Module) :
    def __init__(self, upsample = False, lr=2.1e-3, modules=[
        {'in_channels':3 * 1 * 1, 'kernel_sizes': [3], 'decoder_kernel_size': 3, 'additional_encoder': 1, 'additional_decoder': 1, 'add_innput': True, 'add_fft': 7},
        {'in_channels':3 * 1 * 9, 'kernel_sizes': [3], 'decoder_kernel_size': 3, 'additional_encoder': 1, 'additional_decoder': 1, 'add_innput': True, 'add_fft': 3},
        {'in_channels':3 * 1 * 9, 'kernel_sizes': [3], 'decoder_kernel_size': 3, 'additional_encoder': 0, 'additional_decoder': 0, 'mid_channels': 3 * 1 * 9}
    ]) -> None :
        ## instantiate model + optimizer + loss function + any other stuff you need
        super().__init__()
        scale_factor = 3
        if(upsample):
            in_channels = modules[0]['in_channels']
            modules[0]['in_channels'] = in_channels * scale_factor

        self.model = None
        for module in modules[::-1]:
            self.model = ConvPoolConvTranspose2d(**module, insides = self.model)

        if(upsample):
            self.model = nn.Sequential(
                nn.Upsample(scale_factor = scale_factor),
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = in_channels * scale_factor,
                    kernel_size = scale_factor,
                    stride = scale_factor
                ),
                nn.LeakyReLU(inplace = True),
                self.model,
                nn.Conv2d(
                    in_channels = in_channels * scale_factor,
                    out_channels = in_channels,
                    kernel_size = scale_factor,
                    padding = scale_factor // 2,
                    padding_mode = 'replicate'
                )
            )

        print(self.parameters())
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        self.loss_funct = torch.nn.MSELoss()

        pass

    #show decoded images#########################################
    def show(self, img_1, img_2, img_3):
        with torch.no_grad():
            fig = plt.figure(figsize = (10, 7))

            rows = 1
            columns = 3

            fig.add_subplot(rows, columns, 1)
            plt.imshow(torch.clip(self.denormaliser(img_1), min = 0, max = 255).permute(1, 2, 0))

            fig.add_subplot(rows, columns, 2)
            plt.imshow(torch.clip(self.denormaliser(img_2), min = 0, max = 255).permute(1, 2, 0))

            fig.add_subplot(rows, columns, 3)
            plt.imshow(torch.clip(self.denormaliser(img_3), min = 0, max = 255).permute(1, 2, 0))

            plt.savefig("step.png")
            plt.close(fig)
    #############################################################

    #test method to print input shape############################
    def test(self, input):
        print(input.shape)
        for m in self.model.children():
            input = m(input)
            print(input.shape)
    #############################################################

    #load saved model############################################
    def load_pretrained_model(self, path = f'Miniproject_1{split}bestmodel.pth') -> None :
        ## This loads the parameters saved in bestmodel .pth into the model
        self.load_state_dict(torch.load(path))
        pass
    #############################################################
    
    #Save the model to the path##################################
    def save_model(self, path = 'bestmodel.pth') -> None :
        ## This save the parameters in bestmodel .pth into the model
        torch.save(self.state_dict(), path)
        pass
    #############################################################
    
    #training the model##########################################
    def train(self , train_input , train_target , num_epochs, device = 'cpu', batch_size = 1 << 8, max_time = None, verbose = False, show_step = False) :
        #: train_input : tensor of size(N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size(N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        train_input = self.normaliser(train_input, device)
        train_target = self.normaliser(train_target, device)
        nb = ceil(len(train_input) / batch_size)
        start_time = time()
        for epoch in range(num_epochs):
            if(verbose):
                print(f'epoch {epoch}', end = '\r')
                t = time()
            if((max_time != None) and (epoch > 0) and ((time() - start_time) * (epoch + 1) / epoch) > max_time):
                print((time() - start_time) * (epoch + 1) / epoch)
                break
            for k in range(nb):
                self.optimizer.zero_grad()
                start = k * batch_size
                end = min(start + batch_size, len(train_input))
                output = self.model(train_input[start:end] if(epoch % 2 == 0) else train_target[start:end])

                loss = self.loss_funct(output, train_target[start:end] if(epoch % 2 == 0) else train_input[start:end])
                loss.backward()

                self.optimizer.step()

                if(show_step):
                    s = time()
                    self.show(train_input[start], train_target[start], output[0])
                    s -= time()
                    t -= s
                if(verbose):
                    dt = time() - t
                    print(f'epoch {epoch} {(end*1000 // len(train_input))/10 }% ({round(dt, 2)}s : {round((len(train_input) - end) * dt / end, 2)}s = {round(len(train_input) * dt / end, 2)}s)\t\t\t', end = '\r')

            if(verbose):
                dt = time() - t
                print(f'epoch {epoch} time : {round(dt, 2)}s\t\t\t\t')
        return (epoch + 1, time() - start_time)
    #############################################################

    #forward pass of the model###################################
    def forward(self, input) -> torch.Tensor:
        return self.model(input)
    #############################################################

    #normalise the input image to go from 0-1####################
    def normaliser(self, input, device) -> torch.Tensor:
        with torch.no_grad():
            return (input.float() / 255).to(device)
    #############################################################

    #Denormalise the input to go from 0-255######################
    def denormaliser(self, input) -> torch.Tensor:
        with torch.no_grad():
            return (input * 255).int().to('cpu')
    #############################################################

    #Predict a batch of images###################################
    def predict(self , test_input, device = 'cpu') -> torch.Tensor :
        #: test_input : tensor of size(N1 , C, H, W) that has to be denoised by the trained or the loaded network .
        #: returns a tensor of the size(N1 , C, H, W)
        with torch.no_grad():
            output = self.denormaliser(self.model(self.normaliser(test_input, device)))
            return torch.clip(output, min = 0, max = 255)
    #############################################################