from onmiglot.model import Siamese
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

label_ana = torch.tensor([0], dtype=torch.float32, device='cuda:0').unsqueeze(0)
label_dubravka = torch.tensor([1], dtype=torch.float32, device='cuda:0').unsqueeze(0)
label_uros = torch.tensor([2], dtype=torch.float32, device='cuda:0').unsqueeze(0)

model = Siamese()
# model.load_state_dict(torch.load('/home/wingman2/models/siamese-model.pt'))
model.load_state_dict(torch.load('/home/wingman2/models/model-inter-501.pt'))
model.cuda()
model.eval()
# print(model)

im1 = Image.open("/home/wingman2/datasets/personas/eval/u1.jpg")
im2 = Image.open("/home/wingman2/datasets/personas/eval/u2.jpg")

data_transforms_eval = transforms.Compose([
    # NewPad(),
    transforms.Resize((105, 105)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
    # ,
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

im1 = data_transforms_eval(im1).unsqueeze(0)
im2 = data_transforms_eval(im2).unsqueeze(0)

img1, img2 = Variable(im1.cuda()), Variable(im2.cuda())

output = model.forward(img1, img2)
print(output)

# loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
# real_out = loss_fn(output, label_uros)
# print(real_out)