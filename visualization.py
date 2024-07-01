
# dataset download

'! pip install datasets

from huggingface_hub import login
from datasets import Dataset
login()

from datasets import load_dataset

dataset = load_dataset("raiselab/HUVER_raise", split='train')

index = 3001  # dataset instance to be visulaized 
dataset[3001]



# Image Visulaization

'!pip install pillow

from IPython.display import display

def show_image_from_dataset(image_obj):
  
    display(image_obj)


show_image_from_dataset(dataset[index]['image'])


# 3D mesh visulization

'! pip install datasets pillow trimesh pyglet
'! pip install datasets trimesh httpx

import trimesh
import httpx
from datasets import load_dataset
from IPython.display import display, Image

def load_remote_glb(url, **kwargs):
    """
    Load a remote GLB file using an HTTP GET request and trimesh.
    """
    response = httpx.get(url, follow_redirects=True)
    response.raise_for_status()

   
    if 'model/gltf-binary' not in response.headers['Content-Type']:
        raise ValueError(f"Expected a GLB file but got content type: {response.headers['Content-Type']}")

    
    file_obj = trimesh.util.wrap_as_stream(response.content)
    model = trimesh.load(file_obj, file_type='glb', **kwargs)
    return model

def visualize_glb_model(url):
    """
    Visualize the GLB model from the given URL.
    """
    model = load_remote_glb(url)
    if isinstance(model, trimesh.Scene):
        scene = model
    else:
        scene = trimesh.Scene(model)
    display(scene.show())


entry = dataset[index]
glb_url = entry['glb_file']  

print("Visualizing GLB Model:")
visualize_glb_model(glb_url)

