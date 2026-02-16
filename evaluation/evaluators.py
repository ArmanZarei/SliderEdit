from PIL import Image
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import SiglipProcessor, SiglipModel
from transformers import AutoProcessor, BlipModel, BlipForImageTextRetrieval
from transformers import AutoImageProcessor, AutoModel
import cv2
import lpips
from insightface.app import FaceAnalysis


class VLMEvaluator(ABC):
    @abstractmethod
    def get_score(self, image: Image.Image, text: str) -> float:
        pass


class CLIPEvaluator(VLMEvaluator):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device="cuda"):
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
    
    def get_score(self, image: Image.Image, text: str) -> float:
        inputs = self.clip_processor(text=text, images=image, return_tensors="pt", padding=True)

        inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}

        # Get CLIP embeddings
        outputs = self.clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity
        clip_score = (image_embeds @ text_embeds.T).item()

        return clip_score


class SiglipEvaluator(VLMEvaluator):
    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", device="cuda"):
        self.siglip_model = SiglipModel.from_pretrained(model_name).to(device).eval()
        self.siglip_processor = SiglipProcessor.from_pretrained(model_name)
    
    @torch.inference_mode()
    def get_score(self, image: Image.Image, text: str) -> float:
        inputs = self.siglip_processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.siglip_model.device) for k, v in inputs.items()}

        outputs = self.siglip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds  = outputs.text_embeds

        score = (image_embeds @ text_embeds.T).item()

        return score


class BLIPEvaluator(VLMEvaluator):
    def __init__(self, model_name: str = "Salesforce/blip-itm-base-coco", device="cuda"):
        self.blip_model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device).eval()
        self.blip_processor = AutoProcessor.from_pretrained(model_name)
    

    @torch.inference_mode()
    def get_score(self, image: Image.Image, text: str) -> float:
        inputs = self.blip_processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.blip_model.device) for k, v in inputs.items()}

        outputs = self.blip_model(**inputs)

        itm_score = torch.nn.functional.softmax(outputs.itm_score, dim=1)
        match_prob = itm_score[0, 1].item()

        return match_prob


class FeatureDistanceEvaluator(ABC):
    @abstractmethod
    def get_distance(self, image_1: Image.Image, image_2: Image.Image) -> float:
        pass


class LPIPSFeatureDistanceEvaluator(FeatureDistanceEvaluator):
    def __init__(self, net: str, device="cuda"):
        assert net in ["alex", "vgg"], "Net must be either 'alex' or 'vgg'"

        self.device = device
        self.lpips_model = lpips.LPIPS(net=net).to(device)
    
    def get_distance(self, image_1: Image.Image, image_2: Image.Image) -> float:
        return self.lpips_model(
            lpips.im2tensor(np.array(image_1)).to(self.device),
            lpips.im2tensor(np.array(image_2)).to(self.device)
        ).item()


class DiNOFeatureDistanceEvaluator(FeatureDistanceEvaluator):
    def __init__(self, model_name: str = "facebook/dinov2-base", device="cuda"):
        self.dino_processor = AutoImageProcessor.from_pretrained(model_name)
        self.dino_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
    
    @torch.no_grad()
    def get_dinov2_embed(self, images):
        batch = self.dino_processor(images=images, return_tensors="pt").to(self.dino_model.device)
        out = self.dino_model(**batch)            # last_hidden_state: (N, 1+P, D)
        cls = out.last_hidden_state[:, 0, :]      # take [CLS] token as global descriptor
        cls = F.normalize(cls.float(), dim=-1)    # L2 normalize for cosine sim
        return cls
    
    def get_distance(self, image_1: Image.Image, image_2: Image.Image) -> float:
        feat_1, feat_2 = self.get_dinov2_embed([image_1]), self.get_dinov2_embed([image_2])
        cosine_distance = 1 - feat_1 @ feat_2.T
        
        return cosine_distance.item()


class FaceIdentityDistanceEvaluator(FeatureDistanceEvaluator):
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face_embeddings(self, image: Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = self.app.get(img)
        embs = [f.normed_embedding.astype(np.float32) for f in faces]

        if len(embs) > 1 or len(embs) == 0:
            print(f"Warning: Detected {len(embs)} faces in this image. Expected exactly one face.")
            return None, None

        return np.stack(embs)
    
    def _cosine_similarity(self, a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)

        return np.dot(a_norm, b_norm).item()

    def get_distance(self, image_1: Image.Image, image_2: Image.Image) -> float:
        embs1, embs2 = self.get_face_embeddings(image_1), self.get_face_embeddings(image_2)

        if embs1 is None or embs2 is None:
            return None
        if embs1[0] is None or embs2[0] is None:
            return None

        return 1 - self._cosine_similarity(embs1[0], embs2[0])
