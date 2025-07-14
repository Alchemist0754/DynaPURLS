import torch
import clip

class TextEncoder:
    def __init__(self, cls_labels, bp_num, t_num, device):
        self.cls_labels = cls_labels
        self.bp_num = bp_num
        self.t_num = t_num
        self.device = device
        self.clip, _ = clip.load("ViT-B/32", device=self.device)
        self.full_language = self._generate_text_embeddings()

    def _generate_text_embeddings(self):
        if self.cls_labels is None:
            raise ValueError("cls_labels must be provided")
        
        cls_tokens = [clip.tokenize(self.cls_labels[:, i]) for i in range(self.bp_num + self.t_num + 1)]
        
        with torch.no_grad():
            text_features = []
            for i in range(self.bp_num + self.t_num + 1):
                curr_text_features = self.clip.encode_text(cls_tokens[i].to(self.device)).float()
                curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
                text_features.append(curr_text_features.unsqueeze(0))
            
            text_features = torch.cat(text_features, dim=0)
            text_features = text_features.permute(1, 0, 2).contiguous()  # cls, bp, dim
        
        return text_features

    def get_text_embeddings(self):
        return self.full_language