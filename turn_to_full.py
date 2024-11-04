import torch
import torchvision.models as models
local_path = os.path.dirname(os.path.abspath(__file__))
model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load(local_path + "\\model.pth", weights_only = False))
torch.save(model, local_path + "\\model.pth")