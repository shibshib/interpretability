from collections import OrderedDict
import json
from models import *
import torchvision
import torchvision.transforms as transforms
from models.custom_data_parallel import CustomDataParallel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_batch_size = 1
models = {
    "RegNetX_200MF": RegNetX_200MF(),
    "VGG": VGG('VGG19'),
    "ResNet18": ResNet18(),
    "GoogLeNet": GoogLeNet(),
    "DenseNet121": DenseNet121(),
    "ResNeXt29_2x64d": ResNeXt29_2x64d(),
    "MobileNet": MobileNet(),
    #"MobileNetV2": MobileNetV2(),
    "PreActResNet18": PreActResNet18(),
    #"DPN92": DPN92(),
    "SENet18": SENet18(),
    "EfficientNetB0": EfficientNetB0()
}

# Fast Gradient Sign Attack attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test_fgsm( model, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    loss = None
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # MNIST Test dataset and dataloader declaration
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=True, num_workers=2)
    
    model.eval()
    # Loop over all examples in test set
    for image, target in test_loader:
        # Send the data and label to the device
        image, target = image.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        image.requires_grad = True

        # Forward pass the data through the model
        output = model(image)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        image_grad = image.grad.data

        # Call FGSM Attack
        perturbed_image = fgsm_attack(image, epsilon, image_grad)

        # Re-classify the perturbed image
        output = model(perturbed_image)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex.tolist()) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex.tolist()) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def test_modified_accuracies():
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    results = {}

    map_location = None if torch.cuda.is_available() else 'cpu'
    epsilon = 0
    
    for e in epsilons:
        accuracies = {}
        examples = {}
        print("Processing for e = {}",e)
        for m in models.keys():
            checkpoint_path = "./checkpoint/" + m + ".torch"
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            new_state_dict = OrderedDict()

            model = models[m]
            model = model.to(device)
            model = CustomDataParallel(model)

            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            print("Loaded model {}.".format(m))
            acc, example = test_fgsm(model, e)
            accuracies[m] = acc
            examples[m] = example
            break
        results[e] = {"accuracies": accuracies, "examples": examples}
    return results


epsilons = [0, .05, .1, .15, .2, .25, .3]
results = test_modified_accuracies()
json_str = json.dumps(results)
f = open("results.json","w")
f.write(json_str)
f.close()
