from math import ceil, sqrt
from time import time
import torch
from torch import empty , cat , arange, tensor, exp
from torch.nn.functional import fold , unfold
import ast
import os
split = '\\' if '\\' in os.getcwd() else '/'

SigmaFonctions = []
Optimizers = []
LossFonctions = []

def rand(shape):
    tensor = empty(shape).normal_()
    return tensor.float()

def where(cond, true, false):
    tensor = empty(cond.shape)
    tensor[cond] = true[cond]
    tensor[~cond] = false[~cond]
    return tensor

def zero(shape):
    tensor = empty(shape)
    tensor[tensor!=0] = 0
    return tensor

def ones(shape):
    tensor = empty(shape)
    tensor[tensor!=1] = 1
    return tensor

def clip(input, min = 0, max = 255):
    input[input < min] = min
    input[input > max] = max
    return input

def show(noisy_img, predict_img, clean_img):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 7))

    rows = 1
    columns = 3

    fig.add_subplot(rows, columns, 1)
    plt.imshow(noisy_img.permute(1, 2, 0))

    fig.add_subplot(rows, columns, 2)
    plt.imshow(predict_img.permute(1, 2, 0))

    fig.add_subplot(rows, columns, 3)
    plt.imshow(clean_img.permute(1, 2, 0))
    plt.savefig("step.png")
    
    plt.close(fig)

class Module(object):
    def toTorch(self, val):
        return tensor(val[1]).reshape(val[0])

    def fromTorch(self, val):
        shape = val.shape
        size = 1
        for s in shape:
            size *= s
        result = []
        tmp = val.reshape(size)
        for i in range(size):
            result.append(float(tmp[i]))
        return [list(shape), result]

    def forward_(self, input):
        raise NotImplementedError

    def forward_pass(self, input):
        with torch.no_grad():
            if(type(input) is not tuple):
                input = (input,)
            assert(type(input) is tuple)
            cur_input = input[0]
            output, context = self.forward_(cur_input)
            return (output, context, *input[1:]) if (len(input) > 1) else (output, context)

    def forward(self, input):
        return self.forward_pass(input)[0]

    def backward_(self, gradwrtoutput, context):
        raise NotImplementedError

    def backward_pass(self, gradwrtoutput):
        with torch.no_grad():
            if(type(gradwrtoutput) is not tuple):
                gradwrtoutput = (gradwrtoutput,)
            assert(type(gradwrtoutput) is tuple)
            cur_grad = gradwrtoutput[0]
            cur_context = gradwrtoutput[1]
            gradwrtinput = self.backward_(cur_grad, cur_context)
            return (gradwrtinput, *(gradwrtoutput[2:]))

    def backward(self, input):
        return self.backward_pass(input)[0]

    def normalise(self, input):
        return input

    def denormalise(self, output):
        return output

    def predict(self , input) -> tensor :
        return self.denormalise(self.forward_pass(self.normalise(input))[0])

    def grad(self):
        return zero(1)

    def updatewrtgrad(self, diff):
        pass

    def zero_grad(self):
        pass

    def compute_loss_(self, input, target):
        raise NotImplementedError

    def compute_loss(self, input, target):
        if(type(input) is not tuple):
            input = (input,)
        if(type(target) is not tuple):
            target = (target,)
        assert(type(input) is tuple)
        assert(type(target) is tuple)
        cur_input = input[0]
        cur_target = target[0]
        assert(cur_input.shape == cur_target.shape)
        with torch.no_grad():
            loss = self.compute_loss_(cur_input, cur_target)
        return loss

    def compute_grad_(self, input, target):
        raise NotImplementedError

    def compute_grad(self, input, target):
        if(type(input) is not tuple):
            input = (input,)
        if(type(target) is not tuple):
            target = (target,)
        assert(type(input) is tuple)
        assert(type(target) is tuple)
        cur_input = input[0]
        cur_target = target[0]
        assert(cur_input.shape == cur_target.shape)
        with torch.no_grad():
            grad = self.compute_grad_(cur_input, cur_target)
        return (grad, *input[1:])

    def step_(self, grad, other):
        return -grad, other

    def rec_step(self, grads, others):
        if isinstance(grads, list):
            diffs = []
            new_others = []
            for diff, other in (
                [self.rec_step(grad, None) for grad in grads] 
                if(others is None) else 
                [self.rec_step(grad, other) for (grad, other) in zip(grads, others)]
                ):
                diffs.append(diff)
                new_others.append(other)
            return diffs, new_others
        else:
            diff, other = self.step_(grads, others)
            return diff, other

    def step(self):
        self.updatewrtgrad(self.rec_step(self.grad(), None)[0])

    def train(self , train_input , train_target , num_epochs, batch_size = 1 << 8, loss = None, max_time = None, verbose = False, show_ing = None):
        #: train_input : tensor of size(N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size(N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        if(loss == None):
            loss = self
        
        train_input = self.normalise(train_input)
        train_target = self.normalise(train_target)

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
                self.zero_grad()
                start = k * batch_size
                end = min(start + batch_size, len(train_input))

                output = self.forward_pass(train_input[start:end] if(epoch % 2 == 0) else train_target[start:end])
                
                grad = loss.compute_grad(output, train_target[start:end] if(epoch % 2 == 0) else train_input[start:end])
                
                grad = self.backward_pass(grad)

                self.step()

                if(show_ing is not None):
                    self.save('bestmodel.pth')
                    shape = show_ing[0].shape
                    show(show_ing[0], self.predict(show_ing[0].reshape((1, *shape)))[0], show_ing[1])
                
                if(verbose):
                    dt = time() - t
                    print(f'epoch {epoch} {(end*1000 // len(train_input))/10 }% ({round(dt, 2)}s : {round((len(train_input) - end) * dt / end, 2)}s = {round(len(train_input) * dt / end, 2)}s)\t\t\t', end = '\r')
            if(verbose):
                dt = time() - t
                print(f'epoch {epoch} time : {round(dt, 2)}s\t\t\t\t')
        return (epoch + 1, time() - start_time)

    def load_pretrained_model(self, path = f'Miniproject_2{split}bestmodel.pth') -> None :
        self.load(path)

    def save(self, filename):
        param = self.get_param()

        result = (str(param)
            .replace('], ', '],\n')
            .replace('}', '\n}')
            .replace('}, ', '},\n')
            .replace("[[", "[\n[")
            .replace("[{", "[\n{")
            .replace("]]", "]\n]")
            .replace("}]", "}\n]")
            .replace("['", "[\n'")
            .replace("{'", "{\n'")
            .replace(" '", "\n'")
            .replace('\n\n', '\n')
            .replace(':\n', ':')
        ).split('\n')

        file = ""
        i = 0
        for l in result:
            if(l.endswith('}') or l.endswith(']') or l.endswith('},') or l.endswith('],')):
                i -= 1
            if(l.startswith('[') and (l.endswith(']') or l.endswith('],'))):
                i += 1
            file += ('\t' * i) + l + '\n'
            if(l.startswith('[') and (l.endswith(']') or l.endswith('],'))):
                i -= 1
            if((l.startswith('{') or l.startswith('[') or l.endswith('{') or l.endswith('['))):
                i += 1

        f = open(filename, "w")
        f.write(file)
        f.close()

    def load(self, filename, soft=False):
        f = open(filename, "r")
        param = ast.literal_eval(f.read())
        f.close()
        
        self.set_param(param, soft)

    def get_param(self):
        return {
            'module': 'Module'
        }

    def set_param(self, param, soft=False, curent=True):
        assert('module' in param and (not curent or (param['module'] == 'Module')))

class Conv2d(Module):
    def __init__(self, input_channels, output_channels,
        kernel_size = 3, padding = 0, stride = 1):
        super().__init__()

        #TODO: dilation?

        if(type(kernel_size) is not tuple):
            kernel_size = (kernel_size, kernel_size)
        if(type(padding) is not tuple):
            padding = (padding, padding)
        if(type(stride) is not tuple):
            stride = (stride, stride)

        self.weight = rand((output_channels, input_channels, *kernel_size))
        self.bias = rand((output_channels))
        
        self.gradwrtkernel = zero(self.weight.shape)
        self.gradwrtbias = zero(self.bias.shape)

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        pass

    def compute_output(self, input, kernel, bias):
        assert(input.shape[1] == self.input_channels)

        context = {
            'input' : input,
            'kernel' : kernel,
            'bias' : bias
        }

        outSize = (int(((input.shape[2] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1),
                   int(((input.shape[3] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1))

        uf_input = unfold(input, kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
        shape = uf_input.shape
        uf_input = uf_input.reshape((shape[0], 1, shape[1], shape[2]))

        kernel = unfold(kernel, kernel_size = self.kernel_size)

        output = (uf_input * kernel)\
            .sum(2)

        context |= {'output_shape' : output.shape}
            
        output = output.reshape((input.shape[0], self.output_channels, *outSize))

        output += bias.reshape((1, self.output_channels, 1, 1))

        return output, context

    def compute_gradwrtinput(self, gradwrtoutput, input, kernel, output_shape):
        uf_gradwrtoutput = gradwrtoutput\
            .reshape((output_shape[0], 1, output_shape[1], output_shape[2]))

        kernel = unfold(kernel, kernel_size = self.kernel_size)
        kernel = kernel.permute(1, 0, 2)

        uf_gradwrtoutput = (uf_gradwrtoutput * kernel)\
            .sum(2)
        
        gradwrtinput = fold(uf_gradwrtoutput, input.shape[2:], kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)
        return gradwrtinput

    def compute_gradwrtbias(self, gradwrtoutput, input):
        gradwrtbias = gradwrtoutput.sum((0, 2, 3))
        return gradwrtbias

    def compute_gradwrtkernel(self, gradwrtoutput, input):

        uf_input = unfold(input,
            kernel_size = gradwrtoutput.shape[2:], dilation=self.stride, padding = self.padding)
        shape = uf_input.shape

        uf_input = uf_input\
            .reshape((shape[0], self.input_channels, shape[1]//self.input_channels, shape[2]))

        uf_input = uf_input\
            .permute(1,0,2,3)\
            .reshape((uf_input.shape[1], 1, uf_input.shape[0] * uf_input.shape[2], uf_input.shape[3]))

        uf_gradwrtoutput = unfold(gradwrtoutput,
            kernel_size = gradwrtoutput.shape[2:])

        uf_gradwrtoutput = uf_gradwrtoutput\
            .reshape((uf_gradwrtoutput.shape[0], self.output_channels, uf_gradwrtoutput.shape[1]//self.output_channels, uf_gradwrtoutput.shape[2]))

        uf_gradwrtoutput = uf_gradwrtoutput\
            .permute(1,0,2,3)\
            .reshape((uf_gradwrtoutput.shape[1], uf_gradwrtoutput.shape[0] * uf_gradwrtoutput.shape[2], uf_gradwrtoutput.shape[3]))

        gradwrtkernel = (uf_input * uf_gradwrtoutput)\
            .sum(2)\
            .permute(1,0,2)\
            .reshape(self.weight.shape)

        return gradwrtkernel

    def forward_(self, input):
        return self.compute_output(input, self.weight, self.bias)

    def backward_(self, gradwrtoutput, context):
        assert(gradwrtoutput.shape[1] == self.output_channels)

        input = context['input']
        kernel = context['kernel']
        output_shape = context['output_shape']

        gradwrtinput = self.compute_gradwrtinput(gradwrtoutput, input, kernel, output_shape)

        gradwrtkernel = self.compute_gradwrtkernel(gradwrtoutput, input)

        gradwrtbias = self.compute_gradwrtbias(gradwrtoutput, input)

        self.gradwrtkernel += gradwrtkernel
        self.gradwrtbias += gradwrtbias

        return gradwrtinput

    def grad(self):
        return [self.gradwrtkernel, self.gradwrtbias]

    def updatewrtgrad(self, diff):
        assert(isinstance(diff, list))
        assert(diff[0].shape == self.weight.shape)
        assert(diff[1].shape == self.bias.shape)

        self.weight += diff[0]
        self.bias += diff[1]

    def zero_grad(self):
        self.gradwrtkernel = zero(self.weight.shape)

    def get_param(self):
        return super().get_param() | {
            'module': 'Conv2d',
            'kernel': self.fromTorch(self.weight),
            'bias': self.fromTorch(self.bias),
            'gradwrtkernel': self.fromTorch(self.gradwrtkernel),
            'gradwrtbias': self.fromTorch(self.gradwrtbias),
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'stride': self.stride
        }

    def set_param(self, param, soft=False, curent=True):
        assert('module' in param and (not curent or (param['module'] == 'Conv2d')))
        assert('kernel' in param and list(param['kernel'][0]) == list(self.weight.shape))
        assert('bias' in param and list(param['bias'][0]) == list(self.bias.shape))
        assert('gradwrtkernel' in param and list(param['gradwrtkernel'][0]) == list(self.gradwrtkernel.shape))
        assert('gradwrtbias' in param and list(param['gradwrtbias'][0]) == list(self.gradwrtbias.shape))
        assert('input_channels' in param and param['input_channels'] == self.input_channels)
        assert('output_channels' in param and param['output_channels'] == self.output_channels)
        assert('kernel_size' in param and param['kernel_size'] == self.kernel_size)
        assert('padding' in param and param['padding'] == self.padding)
        assert('stride' in param and param['stride'] == self.stride)
        
        super().set_param(param, soft=soft, curent=False)

        self.weight = self.toTorch(param['kernel'])
        self.bias = self.toTorch(param['bias'])
        self.gradwrtkernel = self.toTorch(param['gradwrtkernel'])
        self.gradwrtbias = self.toTorch(param['gradwrtbias'])

class NearestUpsampling(Module):
    def __init__(self, scale_factor = 2):
        super().__init__()
        self.scale_factor = scale_factor
        pass

    def forward_(self, input):
        output = input\
            .repeat_interleave(self.scale_factor, dim = 2)\
            .repeat_interleave(self.scale_factor, dim = 3)
        return output, input.shape

    def backward_(self, gradwrtoutput, input_shape):
        gradwrtinput = unfold(gradwrtoutput, kernel_size=self.scale_factor, stride=self.scale_factor)\
            .reshape((input_shape[0], input_shape[2], self.scale_factor*self.scale_factor, -1))\
            .sum(2)\
            .reshape(input_shape)
            
        return gradwrtinput

    def get_param(self):
        return super().get_param() | {
            'module': 'NearestUpsampling',
            'scale_factor': self.scale_factor
        }

    def set_param(self, param, soft=False, curent=True):
        assert('module' in param and (not curent or (param['module'] == 'NearestUpsampling')))
        assert('scale_factor' in param and param['scale_factor'] == self.scale_factor)

        super().set_param(param, soft=soft, curent=False)

SigmaFonctions.append('ReLU')
class ReLU(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward_(self, input):
        zero_ = zero(input.shape)
        cond = input < 0
        output = where(cond, zero_, input)
        return output, (cond, zero_)

    def backward_(self, gradwrtoutput, context):
        cond, zero_ = context
        gradwrtinput = where(cond, zero_, gradwrtoutput)
        return gradwrtinput

    def get_param(self):
        return super().get_param() | {
            'module': 'ReLU'
        }

    def set_param(self, param, soft=False, curent=True):
        if(soft):
            assert('module' in param and (not curent or (param['module'] in SigmaFonctions)))
        else:
            assert('module' in param and (not curent or (param['module'] == 'ReLU')))
            
        super().set_param(param, soft=soft, curent=False)

SigmaFonctions.append('LeakyReLU')
class LeakyReLU(Module):
    def __init__(self, alpha = 1e-2):
        super().__init__()
        self.alpha = alpha
        pass

    def forward_(self, input):
        cond = input < 0
        output = where(cond, input * self.alpha, input)
        return output, cond

    def backward_(self, gradwrtoutput, cond):
        gradwrtinput = where(cond, gradwrtoutput * self.alpha, gradwrtoutput)
        return gradwrtinput

    def get_param(self):
        return super().get_param() | {
            'module': 'LeakyReLU',
            'alpha': self.alpha
        }

    def set_param(self, param, soft=False, curent=True):
        if(soft):
            assert('module' in param and (not curent or (param['module'] in SigmaFonctions)))
        else:
            assert('module' in param and (not curent or (param['module'] == 'LeakyReLU')))
            assert('alpha' in param and param['alpha'] == self.alpha)
            
        super().set_param(param, soft=soft, curent=False)

SigmaFonctions.append('Sigmoid')
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward_(self, input):
        exp_ = exp(-clip(input, min=-64, max=64))
        output = 1 /(1 + exp_)
        return output, exp_

    def backward_(self, gradwrtoutput, exp_):
        gradwrtinput = gradwrtoutput * (exp_ / ((1 + exp_)**2))
        return gradwrtinput

    def get_param(self):
        return super().get_param() | {
            'module': 'Sigmoid'
        }

    def set_param(self, param, soft=False, curent=True):
        if(soft):
            assert('module' in param and (not curent or (param['module'] in SigmaFonctions)))
        else:
            assert('module' in param and (not curent or (param['module'] == 'Sigmoid')))
            
        super().set_param(param, soft=soft, curent=False)

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)
        pass

    def forward_pass(self, input):
        if(type(input) is not tuple):
            input = (input,)
        for module in self.modules:
            input = module.forward_pass(input)
        return input

    def backward_pass(self, gradwrtoutput):
        if(type(gradwrtoutput) is not tuple):
            gradwrtoutput = (gradwrtoutput,)
        for module in self.modules[:: - 1]:
            gradwrtoutput = module.backward_pass(gradwrtoutput)
        return gradwrtoutput

    def grad(self):
        return [module.grad() for module in self.modules]

    def updatewrtgrad(self, diffs):
        assert(len(diffs) == len(self.modules))
        for module, diff in zip(self.modules, diffs):
            module.updatewrtgrad(diff)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def get_param(self):
        return super().get_param() | {
            'module': 'Sequential',
            'Sequential': [module.get_param() for module in self.modules],
        }

    def set_param(self, param, soft=False, curent=True):
        assert('module' in param and (not curent or (param['module'] == 'Sequential')))
        assert('Sequential' in param and len(param['Sequential']) == len(self.modules))
        
        super().set_param(param, soft=soft, curent=False)

        for module, p in zip(self.modules, param['Sequential']):
            module.set_param(p, soft)

class Upsampling(Sequential):
    def __init__(self, input_channels, output_channels,
        scale_factor = 2,
        kernel_size = 3):
        super().__init__(
            NearestUpsampling(scale_factor),
            Conv2d(input_channels, output_channels, kernel_size = kernel_size, padding = kernel_size // 2)
        )

    def get_param(self):
        return super().get_param() | {
            'module': 'Upsampling',
        }

    def set_param(self, param, soft=False, curent=True):
        assert('module' in param and (not curent or (param['module'] == 'Upsampling')))
        
        super().set_param(param, soft=soft, curent=False)

    def grad(self):
        return self.modules[1].grad()

    def updatewrtgrad(self, diff):
            self.modules[1].updatewrtgrad(diff)

LossFonctions.append('MSE')
class MSE(Module):
    def __init__(self):
        super().__init__()
        pass

    def compute_loss_(self, input, target):
        return ((input - target) ** 2).mean() / 2

    def compute_grad_(self, input, target):
        return (input - target) / (input.shape[1] * input.shape[2] * input.shape[3])
    
    def get_param(self):
        return super().get_param() | {
            'module': 'MSE'
        }

    def set_param(self, param, soft=False, curent=True):
        if(soft):
            assert('module' in param and (not curent or (param['module'] in LossFonctions)))
        else:
            assert('module' in param and (not curent or (param['module'] == 'MSE')))
            
        super().set_param(param, soft=soft, curent=False)

Optimizers.append('Optimizer')
class Optimizer(Module):
    def __init__(self, model, loss_funct):
        super().__init__()
        self.model = model
        self.loss_funct = loss_funct
        pass

    def forward_pass(self, input):
        return self.model.forward_pass(input)

    def backward_pass(self, gradwrtoutput):
        return self.model.backward_pass(gradwrtoutput)
    
    def compute_loss(self, input, target):
        return self.loss_funct.compute_loss(input, target)

    def compute_grad(self, input, target):
        return self.loss_funct.compute_grad(input, target)

    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        self.model.updatewrtgrad(self.rec_step(self.model.grad(), None)[0])

    def train(self , train_input , train_target , num_epochs, batch_size = 1 << 8, loss = None, max_time = None, verbose = False, show_ing = None):
        super().train(train_input , train_target , num_epochs, batch_size = batch_size, loss = self.loss_funct, max_time = max_time, verbose = verbose, show_ing = show_ing)
        pass

    def get_param(self):
        return super().get_param() | {
            'module': 'Optimizer',
            'model': self.model.get_param(),
            'loss_funct': self.loss_funct.get_param()
        }

    def set_param(self, param, soft=False, curent=True):
        if(soft):
            assert('module' in param and (not curent or (param['module'] in Optimizers)))
        else:
            assert('module' in param and (not curent or (param['module'] == 'Optimizer')))
        assert('model' in param)
        assert('loss_funct' in param)

        super().set_param(param, soft=soft, curent=False)

        self.model.set_param(param['model'], soft)
        self.loss_funct.set_param(param['loss_funct'], soft)

Optimizers.append('BasiqueOptimizer')
class BasiqueOptimizer(Optimizer):
    def __init__(self, model, loss_funct, lr = 1e-4):
        super().__init__(model, loss_funct)
        self.lr = lr
        pass

    def step_(self, grad, other):
        return -self.lr * grad, other
    
    def get_param(self):
        return super().get_param() | {
            'module': 'BasiqueOptimizer',
            'lr': self.lr,
            'model': self.model.get_param(),
            'loss_funct': self.loss_funct.get_param()
        }

    def set_param(self, param, soft=False, curent=True):
        if(soft):
            assert('module' in param and (not curent or (param['module'] in Optimizers)))
        else:
            assert('module' in param and (not curent or (param['module'] == 'BasiqueOptimizer')))
            assert('lr' in param and param['lr'] == self.lr)

        super().set_param(param, soft=soft, curent=False)

Optimizers.append('ADAM')
class ADAM(Optimizer):
    def __init__(self, model, loss_funct, lr = 1e-4, eps = 1e-8, beta1 = 0.9, beta2 = 0.99):
        super().__init__(model, loss_funct)
        self.lr = lr 
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.other = None
        pass

    def step_(self, grad, other):
        mt, vt, t = other if(other is not None) else (zero(grad.shape).float(), zero(grad.shape).float(), 1)
        
        mt = self.beta1*mt + grad*(1-self.beta1)
        vt = self.beta2*vt+(1-self.beta2)*(grad**2)
        
        beta1t = self.beta1**t
        beta2t = self.beta2**t

        alphat = self.lr * sqrt(1-beta2t)/(1-beta1t)
        dif = -alphat*mt/(torch.sqrt(vt)+self.eps)
        t += 1
        other = (mt, vt, t)
        return dif, other

    def step(self):
        grad = self.model.grad()
        dif, self.other = self.rec_step(grad, self.other)
        self.model.updatewrtgrad(dif)
    
    def get_param(self):
        return super().get_param() | {
            'module': 'ADAM',
            'lr': self.lr,
            'eps': self.eps,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'model': self.model.get_param(),
            'loss_funct': self.loss_funct.get_param()
        }

    def set_param(self, param, soft=False, curent=True):
        if(soft):
            assert('module' in param and (not curent or (param['module'] in Optimizers)))
        else:
            assert('module' in param and (not curent or (param['module'] == 'ADAM')))
            assert('lr' in param and param['lr'] == self.lr)
            assert('eps' in param and param['eps'] == self.eps)
            assert('beta1' in param and param['beta1'] == self.beta1)
            assert('beta2' in param and param['beta2'] == self.beta2)

        super().set_param(param, soft=soft, curent=False)


class Model(ADAM):
    def __init__(self):
        model = Sequential(
            Conv2d(input_channels = 3 * 1, output_channels = 3 * 3, stride = 1, kernel_size = 3, padding = 1),
            ReLU(),
            Conv2d(input_channels = 3 * 3, output_channels = 3 * 3, stride = 1, kernel_size = 3, padding = 1),
            ReLU(),
            Conv2d(input_channels = 3 * 3, output_channels = 3 * 1, stride = 1, kernel_size = 3, padding = 1),
            Sigmoid()
        )
        loss_funct = MSE()
        super().__init__(model, loss_funct, lr = 1e-2)
    
    def normalise(self, input):
        return input.float() / 255.0

    def denormalise(self, output):
        return clip(output * 255.0, min = 0, max = 255).int()