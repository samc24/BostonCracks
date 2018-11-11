import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from deeplabv3.model.deeplabv3 import DeepLabV3
from utils import create_sidewalk_segment
from custom_dataloader import ImageFolderWithPaths

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEEPLAB_PRETRAINED_PATH = '/home/devin/Documents/bostonhacks2018/sidewalk_analysis/deeplabv3/pretrained_models/'
BASE_PATH = '/home/devin/Documents/bostonhacks2018/sidewalk_analysis/'
TEST_IMAGES_PATH = '/home/devin/Documents/bostonhacks2018/sidewalk_analysis/gsv_images'

deeplab_model = DeepLabV3('deeplap_1', BASE_PATH).to(DEVICE)

# Apply pretrained deeplab weights
deeplab_model.load_state_dict(
    torch.load(
        DEEPLAB_PRETRAINED_PATH + 'model_13_2_2_2_epoch_580.pth',
        #map_location='cpu'
    )
)

# Create trainset
transform = transforms.Compose([
    transforms.Resize((1024, 2048)),
    transforms.ToTensor(),
])

testset = ImageFolderWithPaths(TEST_IMAGES_PATH, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=2)

# Make predictions
preds_list = []
imgs_list = []
paths_list = []

with torch.no_grad():
    for loaded in testloader:
        imgs = Variable(loaded[0]).to(DEVICE)
        preds = deeplab_model(imgs)
        paths = loaded[2]

        preds_list.append(preds)
        imgs_list.append(imgs)
        paths_list.append(paths)

i = 0
for preds, imgs, paths in zip(preds_list, imgs_list, paths_list):
    create_sidewalk_segment(preds, imgs, BASE_PATH + 'segmented_images/images/', paths_list)
