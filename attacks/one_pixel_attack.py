import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import argparse

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from loguru import logger
from models import *
from utils.differential_evolution import differential_evolution
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {
    "RegNetX_200MF": RegNetX_200MF(),
#    "VGG": VGG('VGG19'),
#    "ResNet18": ResNet18(),
#    "GoogLeNet": GoogLeNet(),
#    "DenseNet121": DenseNet121(),
#    "ResNeXt29_2x64d": ResNeXt29_2x64d(),
#    "MobileNet": MobileNet(),
    #"MobileNetV2": MobileNetV2(),
#    "PreActResNet18": PreActResNet18(),
##    "DPN92": DPN92(),
    #"SENet18": SENet18(),
    #"EfficientNetB0": EfficientNetB0(),
##	  "DLA_simp": SimpleDLA(),
##    "DLA": DLA()	
}

parser = argparse.ArgumentParser(description = "One pixel attack with PyTorch")
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iterations in the DE algorithm')
parser.add_argument('--popsize', default=400, type=int, help='The number of adversarial examples in each iteration')
parser.add_argument('--samples', default=100, type=int, help='Number of image samples to attack')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle')
parser.add_argument('--verbose', action='store_true', help='Print out additional info every iteration')

args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# some hyperparams
test_batch_size = 1

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch_size, shuffle=True, num_workers=2)
    

def perturb_image(xs, img):
    """Perturbs a given image along a given axis.

    Args:
        xs (range func): axis along which to perturb image.
        img (tensor): test image to be perturbed.
    """
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x)/5)

        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
			imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
			imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
        count += 1

    return imgs

def predict_classes(xs, img, target_class, net, minimize=True):
    """ 

    Args:
        xs (range func): axis along which to perturb image.
        img (tensor): test image to be perturbed.
        target_class (int): class corresponding to img.
        net (nn.module): Model on which to perform this attack
        minimize (bool, optional): Defaults to True.
    """
    imgs_perturbed = perturb_image(xs, img.clone())
    input = Variable(imgs_perturbed, volatile=True).to(device)
    predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_class]

    return predictions if minimize else 1 - predictions


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
    """ Performs attack on a given image

    Args:
        img (tensor): 1 * 3 * W * H tensor
        label (int): integer representation of class label.
        net (nn.module): Model on which to perform this attack.
        target (int, optional): Target class.
        pixels (int, optional): number of pixels that can be perturbed. Defaults to 1.
        maxiter (int, optional): maximum number of iteration in the DE algorithm. Defaults to 75.
        popsize (int, optional): maximum number of iteration in the DE algorithm. Defaults to 400.
        verbose (bool, optional): print maximum number of iteration in the DE algorithm. Defaults to False.
    """

    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels
    popmul = max(1, popsize/len(bounds))

    predict_fn = lambda xs: predict_classes(xs, img, target_class, net, target is None)
    callback_fn = lambda x, convergence: attack_success(x, img, target_class, net, targeted_attack, verbose)

    inits = np.zeros([popmul * len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i*5+0] = np.random.random()*32
            init[i*5+1] = np.random.random()*32
            init[i*5+2] = np.random.normal(128,127)
			init[i*5+3] = np.random.normal(128,127)
			init[i*5+4] = np.random.normal(128,127)
    
    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul, recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)
    attack_image = perturb_image(attack_result.x, img)
    attack_var = Variable(attack_image, volatile=True).to(device)
    predicted_probs = F.softmax(net(attack_var)).data.cpu().numpy()[0]

    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int)
    
    return 0, [None]



 def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):
    correct=0
    success=0

    for batch_idx, (input, target) in enumerage(loader):
        img_var = Variable(input, volatile=True).to(device)
        prior_probs = F.softmax(net(img_var))
        _, indices = torch.max(prior_probs, 1)

        if target[0] != indices.data.cpu()[0]:
            continue
        
        correct += 1
        target = target.numpy()
        targets = [None] if not targeted else range(10)

        for target_class in targets:
            if targeted:
                if target_class == target[0]:
                    continue
            
            flag, x = attack(input, target[0], net, target_class, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)
            success += flag
            



if __name__ == "__main__":
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    map_location = None if torch.cuda.is_available() else 'cpu'
    for m in models.keys():
            logger.info("Loading data and {} model ... ".format(m))
            checkpoint_path = "../checkpoint/" + m + ".torch"
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            net = models[m].to(device)
            
            logger.info("Starting attack on Model {}...".format(m))
            #results = attack_all(net, testloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
            
