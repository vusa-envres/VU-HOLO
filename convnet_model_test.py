import numpy as np
import torch 
from models.convnet import ConvNet
from PIL import Image
import skimage
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from matplotlib.path import Path

iScale = 2**15
fScale = float(iScale)

Labels = ['Corylus',        #0
          'Alnus',          #1
          'Betula',         #2
          'Carpinus',       #3
          'Salix',          #4
          'Populus',        #5
          'Ulmus',          #6
          'Cupresud'        #7
          'Fraxinus',       #8
          'Pinus S',        #9
          'Pinus_M',        #10
          'Picea',          #11
          'Quercus',        #12
          'Poceace',        #13
          'Plantago',       #14
          'Chenopodium',    #15
          'Artemisia',      #16
          'Ambrosia_art',   #17
          'Iva',            #18
          'Humulus',        #19
          'Acer',           #20
          'Alopecueus',     #21
          'Urtica',         #22
]

def contour_perimeter(contour):
    # Ensure the contour is closed by appending the first point at the end
    closed_contour = np.vstack([contour, contour[0]])
    # Compute the Euclidean distances between successive points
    distances = np.sqrt(np.sum(np.diff(closed_contour, axis=0) ** 2, axis=1))
    return np.sum(distances)

def prepare_image(filename):
    gray_image = np.array(Image.open(filename)) 
    denoised_image = gaussian(gray_image, sigma=2)
    thresh = threshold_otsu(denoised_image)
    binary_image = denoised_image < thresh
    binary = np.zeros(shape=binary_image.shape)
    binary[binary_image] = 1
    contours = find_contours(binary, level=0.8)
    for c in contours:
        p = contour_perimeter(c)
        if p < 40:
            path = Path(c)
            x, y = np.meshgrid(np.arange(binary_image.shape[1]), np.arange(binary_image.shape[0]))
            points = np.vstack((x.flatten(), y.flatten())).T
            inside = path.contains_points(points)
            mask = inside.reshape(binary_image.shape)
            binary[mask.T] = np.invert(binary[mask.T]==1 )
    contours = find_contours(binary, level=0.8)
    size = np.sum(binary)
    x, y = skimage.measure.centroid(binary)  
    ix = int(x)
    iy = int(y)
    image = gray_image[ix-50:ix+50,iy-50:iy+50]    
    nc = len(contours)
    return image, nc, size
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_state = torch.load("checkpoints/model_conv.pth")
model = ConvNet()
model.load_state_dict(model_state)
model.to(device)
model.eval()


f0 = "poleno_image_rec0.png"
f1 = "poleno_image_rec1.png"

image0, nc0, size0 = prepare_image(f0)
image1, nc1, size1 = prepare_image(f1) 

if nc0==1 and nc1==1:
    try:
        x0 = (image0 - iScale) / fScale
        x1 = (image1 - iScale) / fScale

        tensor0 = torch.from_numpy(x0).float().unsqueeze(0).unsqueeze(0).to(device)
        tensor1 = torch.from_numpy(x1).float().unsqueeze(0).unsqueeze(0).to(device)
        outputs = model(tensor0, tensor1)
        conf, predicted = torch.max(outputs.data, 1)
        print(Labels[predicted])
    except:
        print('Recognition error')
