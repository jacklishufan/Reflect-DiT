import torch.nn.functional as F
import torch
from typing import List,Optional

@torch.no_grad()
def encode_image_pool(image: torch.Tensor,image_encoder, n_query=64):
    n_query = n_query 

    image_embeds = image_encoder(image).last_hidden_state # 
    #image_embeds = image_embeds[:, 1:, :]
    b, n, c = image_embeds.shape
    sqrt_n = int(n**0.5)
    assert sqrt_n ** 2 == n
    image_embeds = image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)

    stride = int(sqrt_n // (n_query ** 0.5))
    image_embeds = F.avg_pool2d(image_embeds, kernel_size=(stride, stride), stride=stride)
    image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1).contiguous() # N L C
    return image_embeds


def pad_reflection(reflection_context_list: List[torch.Tensor],reflection_context_length: Optional[List[int]]= None):
    if reflection_context_length is None:
        reflection_context_length = torch.tensor(list([len(x) for x in reflection_context_list]))
    # if isinstance(reflection_context_length,torch.Tensor):
    #     reflection_context_length = reflection_context_length.cpu().numpy()
    bsz = len(reflection_context_list)
    max_len = int(max(reflection_context_length))
    dim = reflection_context_list[0].shape[-1]
    device = reflection_context_list[0].device
    # output = torch.zeros(
    #     (bsz,max_len,dim)
    # )
    final_context_list = []
    attention_mask = torch.zeros(bsz,max_len,device=device)
    for idx,(context, length) in enumerate(zip(reflection_context_list,reflection_context_length)):
        num_pad_length = max_len - length
        context_padded = torch.cat(
            [context,torch.zeros(num_pad_length,dim).to(context)],dim=0
        )
        final_context_list.append(context_padded)
        attention_mask[idx,:length] = 1
    final_context = torch.stack(final_context_list)
    assert final_context.shape == (bsz,max_len,dim),f'Expected {(bsz,max_len,dim)}, but found {final_context.shape}'
    return final_context,attention_mask


if __name__ == '__main__':
    from transformers import CLIPImageProcessor,SiglipVisionModel  
    clip  = 'google/siglip-large-patch16-384'
    image_encoder = SiglipVisionModel.from_pretrained(clip)
    image_encoder.cuda()
    x = torch.rand(1,3,384,384).cuda()
    y = image_encoder(x)
    
